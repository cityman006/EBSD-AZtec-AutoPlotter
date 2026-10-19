"""Out-of-core feature building and symmetry-aware KAM and boundary categorisation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import h5py
from .crystal import quat_xyzw, misorientation_deg, ipf_colors
from .storage import transactional_output, check_cancelled

CHANNELS=('qx','qy','qz','qw','phase_id','kam_deg','band_contrast')

def _write_rows(dset,first,stop,data):
    dset[first:stop]=data

@transactional_output()
def prepare(src, outdir, batch=48, do_ipf=True, progress=print, compression='lzf'):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    fpath=outdir/'features.h5'
    if fpath.exists():
        raise FileExistsError(f'Refusing to overwrite existing analysis cache: {fpath}')
    workpath=fpath  # The output transaction already owns a private staging directory.
    meta=src.header|{'angular_units_input':src.probe_euler_unit(), 'channel_names':CHANNELS,
                     'ipf_convention':'orix TSL, NOT Oxford AZtec validated',
                     'kam_convention':'4-neighbour; same-phase; 5-degree neighbour cutoff; 2-degree grain-boundary threshold'}
    metatmp=outdir/'metadata.json'
    metatmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    h,w=src.ny,src.nx
    ky=min(batch,h);kx=min(256,w)
    with h5py.File(workpath,'x') as f:
        kw={'chunks':(1,ky,kx),'compression':compression}
        ds=f.create_dataset('features',shape=(7,h,w),dtype='f4',**kw)
        mask=f.create_dataset('indexed',shape=(h,w),dtype='u1',chunks=(ky,kx),compression=compression)
        f.create_dataset('kam_valid',shape=(h,w),dtype='u1',chunks=(ky,kx),compression=compression)
        f.create_dataset('boundary_right',shape=(h,w),dtype='u1',chunks=(ky,kx),compression=compression)
        f.create_dataset('boundary_down',shape=(h,w),dtype='u1',chunks=(ky,kx),compression=compression)
        f.create_dataset('angle_right_deg',shape=(h,w),dtype='f2',chunks=(ky,kx),compression=compression)
        f.create_dataset('angle_down_deg',shape=(h,w),dtype='f2',chunks=(ky,kx),compression=compression)
        f.attrs['channels']=json.dumps(CHANNELS)
        f.attrs['metadata_json']=json.dumps(meta)
        # pass 1 quat, phase, BC, and optional per-field quality maps
        for field in ('pq','mad'):
            if src.available_fields.get(field,False):
                f.create_dataset(field,shape=(h,w),dtype='f4',chunks=(ky,kx),compression=compression)
        if 'mad' in f:
            f['mad'].attrs['Unit']=src.header['mad_unit']
            f['mad'].attrs['unit_evidence_json']=json.dumps(src.header['mad_unit_evidence'])
            f['mad'].attrs['storage']='unchanged source values; convert only for display'
        for a,b,block in src.iter_rows(batch=batch,fields=('euler','phase','bc','pq','mad')):
            check_cancelled()
            ph=block['phase']; good=(ph>0)
            q=np.zeros((b-a,w,4),dtype='float32')
            if good.any():
                raw=block['euler'][good]
                q[good]=quat_xyzw(np.deg2rad(raw) if meta['angular_units_input']=='degree' else raw)
            ds[0:4,a:b,:]=np.moveaxis(q,-1,0)
            ds[4,a:b,:]=ph.astype('float32')
            ds[6,a:b,:]=block.get('bc',np.zeros_like(ph))
            mask[a:b,:]=good
            for field in ('pq','mad'):
                if field in block:f[field][a:b,:]=block[field]
            if (a//batch)%12==0:progress(f'features: {b}/{h} rows')
        # pass 2 KAM and grain-boundary candidate edges, with 1-row halo per batch
        for a in range(0,h,batch):
            check_cancelled()
            b=min(a+batch,h)
            top=max(0,a-1);bottom=min(h,b+1)
            q=np.moveaxis(ds[0:4,top:bottom,:],0,-1)
            p=np.rint(ds[4,top:bottom,:]).astype('uint8')
            qcenter=q[a-top:b-top];pc=p[a-top:b-top]
            s=np.zeros((b-a,w),dtype='float32')
            cnt=np.zeros((b-a,w),dtype='uint8')
            def absorb(q1,q2,p1,p2,at):
                angles=misorientation_deg(q1,q2,p1,p2,src.phase_specs)
                valid=np.isfinite(angles)&(angles<=5)
                s[at]+=np.where(valid,angles,0)
                cnt[at]+=valid.astype('uint8')
                return angles
            absorb(qcenter[:,1:],qcenter[:,:-1],pc[:,1:],pc[:,:-1],(slice(None),slice(1,None)))
            right=absorb(qcenter[:,:-1],qcenter[:,1:],pc[:,:-1],pc[:,1:],(slice(None),slice(None,-1)))
            if a>0:
                # whole block has an upper neighbour (including previous chunk)
                absorb(qcenter,q[a-top-1:b-top-1],pc,p[a-top-1:b-top-1],(slice(None),slice(None)))
            elif b-a>1:
                absorb(qcenter[1:],qcenter[:-1],pc[1:],pc[:-1],(slice(1,None),slice(None)))
            if b<h:
                absorb(qcenter,q[a-top+1:b-top+1],pc,p[a-top+1:b-top+1],(slice(None),slice(None)))
            elif b-a>1:
                absorb(qcenter[:-1],qcenter[1:],pc[:-1],pc[1:],(slice(None,-1),slice(None)))
            kval=np.divide(s,cnt,out=np.zeros_like(s),where=cnt>0)
            ds[5,a:b,:]=kval
            f['kam_valid'][a:b,:]=(cnt>0).astype('uint8')
            # boundary categories: 0-none 1-LAGB 2-HAGB 3-interphase 255-unknown
            br=np.zeros((b-a,w),dtype='uint8')
            bd=np.zeros((b-a,w),dtype='uint8')
            def edge_category(ang,p1,p2):
                x=np.zeros(p1.shape,dtype='uint8')
                x[(p1==0)|(p2==0)]=255
                x[(p1>0)&(p2>0)&(p1!=p2)]=3
                same=(p1>0)&(p1==p2)&np.isfinite(ang)
                x[same&(ang>=2)&(ang<15)]=1
                x[same&(ang>=15)]=2
                return x
            br[:,:-1]=edge_category(right,pc[:,:-1],pc[:,1:])
            br[:,-1]=255
            down=np.full((b-a,w),np.nan,dtype='float32')
            if b<h:
                down=misorientation_deg(qcenter,q[a-top+1:b-top+1],pc,p[a-top+1:b-top+1],src.phase_specs)
                bd[:,:]=edge_category(down,pc,p[a-top+1:b-top+1])
            elif b-a>1:
                dd=misorientation_deg(qcenter[:-1],qcenter[1:],pc[:-1],pc[1:],src.phase_specs)
                down[:-1]=dd
                bd[:-1]=edge_category(dd,pc[:-1],pc[1:])
            bd[-1,:]=255 if b==h else bd[-1,:]
            f['boundary_right'][a:b]=br
            f['boundary_down'][a:b]=bd
            ar=np.full((b-a,w),np.nan,dtype='float16')
            ar[:,:-1]=right.astype('float16')
            f['angle_right_deg'][a:b]=ar
            f['angle_down_deg'][a:b]=down.astype('float16')
            if (a//batch)%12==0:progress(f'KAM/boundaries: {b}/{h} rows')
    if do_ipf:
        with h5py.File(workpath,'a') as f:
            for ax in 'XYZ':
                d=f.create_dataset('ipf_'+ax,shape=(h,w,3),dtype='u1',chunks=(ky,kx,3),compression=compression)
                for a in range(0,h,batch):
                    check_cancelled()
                    b=min(a+batch,h)
                    block=src.read(a,b,('euler','phase'))
                    e=block['euler']
                    if meta['angular_units_input']=='degree':e=np.deg2rad(e)
                    d[a:b,:,:]=ipf_colors(e,block['phase'],src.phase_specs,axis=ax)
                    if (a//batch)%12==0:progress(f'IPF-{ax}: {b}/{h} rows')
    # Publish only the complete cache. The run directory is private and the
    # destination is required to be absent, so an old history entry is safe.
    # The output transaction validates and atomically publishes these files.
    progress('Saved: '+str(fpath))
    return fpath
