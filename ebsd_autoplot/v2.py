"""V2: ROI grain reconstruction, GOS/GROD, boundary stats, texture, BOR candidates.

This is *exploratory* grain reconstruction based on nearest-neighbour edges;
requires external MTEX/AZtec validation for scientific publication.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from .crystal import misorientation_deg, _sym_for_spacegroup
from .storage import transactional_output, check_cancelled

try:
    from numba import njit
except ImportError:
    njit=None

if njit is not None:
    @njit
    def _root(par,x):
        while par[x]!=x:
            par[x]=par[par[x]]
            x=par[x]
        return x
    @njit
    def _union(par,a,b):
        ra=_root(par,a); rb=_root(par,b)
        if ra!=rb:
            if ra<rb:par[rb]=ra
            else:par[ra]=rb
    @njit
    def _components_nb(valid, ar, ad, threshold):
        h,w=valid.shape;n=h*w
        par=np.arange(n,dtype=np.int32)
        for r in range(h):
            for c in range(w):
                if not valid[r,c]:continue
                k=r*w+c
                if c+1<w and valid[r,c+1] and np.isfinite(ar[r,c]) and ar[r,c]<=threshold:
                    _union(par,k,k+1)
                if r+1<h and valid[r+1,c] and np.isfinite(ad[r,c]) and ad[r,c]<=threshold:
                    _union(par,k,k+w)
        labels=np.zeros(n,dtype=np.int32)
        for r in range(h):
            for c in range(w):
                if valid[r,c]:labels[r*w+c]=_root(par,r*w+c)+1
        return labels.reshape((h,w))


def _components_python(valid,ar,ad,threshold):
    if valid.size>300000:
        raise RuntimeError('Install numba to reconstruct >300,000 pixels: py -m pip install numba')
    h,w=valid.shape
    parent=np.arange(valid.size,dtype=np.int32)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for y in range(h):
        for x in range(w):
            if not valid[y,x]:continue
            a=y*w+x
            if x+1<w and valid[y,x+1] and np.isfinite(ar[y,x]) and ar[y,x]<=threshold:
                parent[root(a+1)]=root(a)
            if y+1<h and valid[y+1,x] and np.isfinite(ad[y,x]) and ad[y,x]<=threshold:
                parent[root(a+w)]=root(a)
    result=np.zeros((h,w),np.int32)
    for y,x in zip(*np.where(valid)):
        result[y,x]=root(y*w+x)+1
    return result


def _mean_quat_sym(q,space_group,iterations=3):
    """Iterated symmetry-aligned Markley-esque quaternion estimate on one grain.

    Returns SciPy xyzw convention; unweighted mean of representative quaternions.
    """
    sy=_sym_for_spacegroup(space_group)
    q=q.astype(np.float64)
    q/=np.linalg.norm(q,axis=1,keepdims=True).clip(min=1e-12)
    ref=q[0].copy()
    for _ in range(iterations):
        means=np.zeros(4,np.float64)
        for k in range(0,len(q),4000):
            part=q[k:k+4000]
            # equivalent q*s orientation; maximize dot to current reference
            r=Rotation.from_quat(part)
            # q*s symmetry-aligned variants: quaternion multiplication
            aa=part[:,None,:]; bb=sy[None,:,:]
            av=aa[...,:3]; aw=aa[...,3:4]; bv=bb[...,:3]; bw=bb[...,3:4]
            vv=aw*bv+bw*av+np.cross(av,bv)
            ww=aw*bw-np.sum(av*bv,axis=-1,keepdims=True)
            eq=np.concatenate([vv,ww],axis=-1)
            dots=np.sum(eq*ref[None,None,:],axis=-1)
            idx=np.argmax(np.abs(dots),axis=1)
            aligned=eq[np.arange(len(part)),idx]
            aligned*=np.where(dots[np.arange(len(part)),idx]<0,-1.,1.)[:,None]
            means+=aligned.sum(axis=0)
        means/=np.linalg.norm(means).clip(min=1e-12)
        if np.abs(np.dot(means,ref))>1-1e-10:
            ref=means;break
        ref=means
    return ref.astype(np.float32)


def bor_variant_angles():
    """Ideal Burgers OR alpha-alpha variant pair angle catalogue.

    Geometry: (110)beta || (0001)alpha; [1,-1,1]beta lies in (110)?
    A validated right-handed orthobasis is built from beta plane normal and
    beta in-plane <111>. Returns unique child-child disorientation values.
    """
    beta_n=np.array([1.,1.,0.]); beta_n/=np.linalg.norm(beta_n)
    beta_d=np.array([1.,-1.,1.]);beta_d/=np.linalg.norm(beta_d)
    assert abs(beta_d@beta_n)<1e-9
    beta_y=np.cross(beta_n,beta_d)
    # columns are alpha X,Y,Z directions expressed in beta crystal coordinates
    R=np.column_stack([beta_d,beta_y,beta_n]); assert np.linalg.det(R)>0.999
    B=Rotation.from_matrix(R)
    cub=Rotation.create_group('O')
    variants=[g*B for g in cub]
    sy=_sym_for_spacegroup(194)
    from .crystal import misorientation_deg
    fake={1:{'space_group':194}}
    uniq=[]
    for v in variants:
        q=v.as_quat().astype(np.float32)
        if not uniq or np.all(misorientation_deg(np.array([q]*len(uniq)),np.array(uniq),np.ones(len(uniq),np.uint8),np.ones(len(uniq),np.uint8),fake)>0.02):
            uniq.append(q)
    ua=np.asarray(uniq)
    if len(ua)<2:return []
    ang=[]
    for i in range(len(ua)):
        for j in range(i+1,len(ua)):
            v=float(misorientation_deg(ua[i:i+1],ua[j:j+1],np.array([1]),np.array([1]),fake)[0])
            if v>0.01:ang.append(v)
    return sorted(set(round(v,2) for v in ang))


@transactional_output()
def run_v2(features_file,outdir,roi=None,threshold=5.,min_size=8,max_pixels=2200000,progress=print):
    outdir=Path(outdir);outdir.mkdir(exist_ok=True,parents=True)
    with h5py.File(features_file,'r') as f:
        h,w=f['indexed'].shape
        if roi is None:
            if h*w>max_pixels:
                raise ValueError(f'Full scan {w}x{h} exceeds V2 safety threshold ({max_pixels} pixels). '
                                 'Use --roi X Y WIDTH HEIGHT; V1 full-field remains supported.')
            x=y=0;rw=w;rh=h
        else:
            x,y,rw,rh=map(int,roi)
            if min(x,y)<0 or min(rw,rh)<1 or x+rw>w or y+rh>h:raise ValueError('ROI outside scan')
            if rw*rh>max_pixels:raise ValueError(f'ROI > {max_pixels} pixels. Use smaller ROI or increase max_pixels if RAM permits')
        sl=(slice(y,y+rh),slice(x,x+rw))
        phase=f['features'][4,sl[0],sl[1]].astype(np.uint8)
        q=np.moveaxis(f['features'][0:4,sl[0],sl[1]],0,-1).copy()
        ar=f['angle_right_deg'][sl].astype('float32')
        ad=f['angle_down_deg'][sl].astype('float32')
        valid=phase>0
        check_cancelled()
        if njit is not None: raw=_components_nb(valid,ar,ad,float(threshold))
        else:raw=_components_python(valid,ar,ad,float(threshold))
        check_cancelled()
        unique,inv=np.unique(raw,return_inverse=True)
        ids=inv.reshape(raw.shape).astype(np.int32)
        ids[~valid]=0
        # Remap indexes so invalid=0 and valid IDs start at 1
        vraw=raw[valid]
        uniq=np.unique(vraw)
        flat=np.searchsorted(uniq,vraw)+1
        ids[:]=0;ids[valid]=flat
        n=len(uniq)
        counts=np.bincount(ids.ravel(),minlength=n+1)
        samples=np.flatnonzero(valid.ravel())
        sums_x=np.bincount(ids.ravel()[samples],weights=samples%rw,minlength=n+1)
        sums_y=np.bincount(ids.ravel()[samples],weights=samples//rw,minlength=n+1)
        mean_x=sums_x/np.maximum(counts,1)
        mean_y=sums_y/np.maximum(counts,1)
        # iterate pixel labels efficiently by stable sort (not per-grain full scan)
        sorted_pix=np.argsort(ids.ravel(),kind='stable')
        boundaries=np.cumsum(counts)
        rows=[]
        grod=np.full((rh,rw),np.nan,np.float32)
        min_size=int(min_size)
        for gid in range(1,n+1):
            check_cancelled()
            pix=sorted_pix[boundaries[gid-1]:boundaries[gid]]
            # because ids=0 invalid, boundaries[0] marks end of zeros
            if len(pix)<min_size:continue
            pid=int(phase.ravel()[pix[0]])
            # extract q and compute mean; protect against huge individual grains via stride
            q0=q.reshape(-1,4)[pix]
            if len(q0)>100000:
                q_sample=q0[np.linspace(0,len(q0)-1,100000,dtype=int)]
            else:q_sample=q0
            # only HCP / BCC
            specs=json.loads(f.attrs['metadata_json'])['phase_specs']
            spec=specs.get(str(pid),specs.get(pid))
            sg=int(spec['space_group'])
            avg=_mean_quat_sym(q_sample,sg)
            a=misorientation_deg(q0,np.broadcast_to(avg,q0.shape),np.full(len(pix),pid,np.uint8),np.full(len(pix),pid,np.uint8),{pid:{'space_group':sg}})
            grod.ravel()[pix]=a
            gos=float(np.mean(a))
            step_um=json.loads(f.attrs['metadata_json'])['x_step_um']
            stepy_um=json.loads(f.attrs['metadata_json'])['y_step_um']
            area=float(len(pix)*step_um*stepy_um)
            eqdiam=float(2*np.sqrt(area/np.pi))
            rows.append((gid,pid,len(pix),area,eqdiam,mean_x[gid]+x,mean_y[gid]+y,gos,*avg))
            if len(rows)%1500==0:progress(f'V2 statistics: {len(rows)}/{n} grains >= min_size')
        import csv
        csvpath=outdir/'grains.csv'
        with csvpath.open('w',newline='',encoding='utf-8') as fd:
            writer=csv.writer(fd)
            writer.writerow(['grain_id','phase_id','pixels','area_um2','eq_diameter_um','centroid_x_px','centroid_y_px','GOS_deg','mean_qx','mean_qy','mean_qz','mean_qw'])
            writer.writerows(rows)
        with h5py.File(outdir/'v2_grains.h5','w') as out:
            from .storage import digest
            out.attrs['source_features_sha256']=digest(features_file)
            out.create_dataset('grain_id',data=ids,compression='lzf')
            out.create_dataset('GROD_deg',data=grod,compression='lzf')
            out.attrs['roi_xywh']=roi if roi else (0,0,w,h)
            out.attrs['threshold_deg']=threshold
        # edge categories and adjacency graph counts, including phase boundaries
        boundary_stats={}
        for label,value in [('low_angle',1),('high_angle',2),('interphase',3),('unknown',255)]:
            cnt=0
            for key in ['boundary_right','boundary_down']:
                val=f[key][sl]
                cnt+=int(np.count_nonzero(val==value))
            boundary_stats[label+'_pixel_edges']=cnt
        # graph edges in grain_id map, weighted by pixel-edge counts
        edges={}
        for aa,bb in [(ids[:,:-1],ids[:,1:]),(ids[:-1,:],ids[1:,:])]:
            mism=(aa!=bb)&(aa>0)&(bb>0)
            for a_id,b_id in zip(aa[mism].tolist(),bb[mism].tolist()):
                pair=(min(a_id,b_id),max(a_id,b_id))
                edges[pair]=edges.get(pair,0)+1
        import csv
        # Experimental alpha-variant pair flags based on ideal BOR disorientation catalogue.
        phase_metadata=json.loads(f.attrs['metadata_json'])['phase_specs']
        alpha_ids={int(pid) for pid,spec in phase_metadata.items() if int(spec.get('space_group',0))==194
                   and any(t in str(spec.get('name','')).casefold() for t in ('alpha','α','hcp','titanium-hex'))}
        catalogue=bor_variant_angles() if any(r[1] in alpha_ids for r in rows) else []
        indexed={r[0]:r for r in rows}
        with (outdir/'grain_neighbour_graph.csv').open('w',encoding='utf-8',newline='') as fd:
            wr=csv.writer(fd)
            wr.writerow(['grain_a','grain_b','shared_pixel_edges','alpha_alpha_misorientation_deg',
                         'nearest_BOR_variant_pair_angle_deg','BOR_candidate_within_2deg'])
            for (a,b),c in edges.items():
                mis=ref=np.nan;match=0
                if a in indexed and b in indexed and indexed[a][1] in alpha_ids and indexed[a][1]==indexed[b][1] and catalogue:
                    phase_id=int(indexed[a][1])
                    aa=np.array([indexed[a][8:12]],np.float32)
                    bb=np.array([indexed[b][8:12]],np.float32)
                    mis=float(misorientation_deg(aa,bb,np.array([phase_id],np.uint8),np.array([phase_id],np.uint8),{phase_id:{'space_group':194}})[0])
                    ref=float(min(catalogue,key=lambda ang:abs(ang-mis)))
                    match=int(abs(mis-ref)<=2.)
                wr.writerow([a,b,c,mis,ref,match])
        # experimental BOR angles, not a definitive prior-beta reconstruction
        parents={'method':'BOR alpha-alpha variant-pair catalogue (screening only, NOT parent-beta reconstruction)',
                 'ideal_variant_pair_angles_deg':catalogue,
                 'status':'candidate-only: original beta grain reconstruction requires iterative parent orientation fitting and validation'}
        summary={'roi':[x,y,rw,rh],'grain_threshold_deg':threshold,'minimum_area_pixels':min_size,
                 'valid_pixels':int(valid.sum()),'reconstructed_components':n,'reported_grains':len(rows),
                 'boundary_counts':boundary_stats,'bor':parents,
                 'warning':'GOS/GROD and grain reconstruction are exploratory; validate against MTEX/AZtec with identical cleaning parameters.'}
        (outdir/'v2_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        # research figure for V2 maps
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap
        fig,ax=plt.subplots(1,3,figsize=(13,4.2),dpi=170,layout='constrained')
        col=ids.astype('uint32')*np.uint32(2654435761)
        rgb=np.stack([(col>>16)&255,(col>>8)&255,col&255],axis=-1).astype('u1')
        rgb[~valid]=0
        ax[0].imshow(rgb,interpolation='nearest');ax[0].set_title('Reconstructed grain IDs (ROI)')
        raw_kam=f['features'][5,sl[0],sl[1]]
        ik=ax[1].imshow(raw_kam,vmin=0,vmax=5,cmap='jet',interpolation='nearest')
        ax[1].set_title('KAM (°)');fig.colorbar(ik,ax=ax[1],fraction=0.046)
        ig=ax[2].imshow(grod,vmin=0,vmax=10,cmap='magma',interpolation='nearest')
        ax[2].set_title('GROD (°) | symmetry-aligned mean');fig.colorbar(ig,ax=ax[2],fraction=0.046)
        for aa in ax:aa.axis('off')
        fig.suptitle('V2 exploratory ROI analysis | original β not reconstructed; verify MTEX',fontsize=10)
        fig.savefig(outdir/'V2_grain_KAM_GROD.png');plt.close(fig)
        progress('V2 ROI analysis complete')
        return summary
