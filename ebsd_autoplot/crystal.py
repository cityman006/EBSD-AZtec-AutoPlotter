"""Symmetry-aware crystal quantities (Bunge ZXZ, radians).

IPF requires orix. Missing orix causes an explicit error, NEVER fake Euler RGB.
"""
from __future__ import annotations
import numpy as np
from .storage import transactional_output, check_cancelled
from scipy.spatial.transform import Rotation


def quat_xyzw(euler_rad):
    shape=euler_rad.shape[:-1]
    return Rotation.from_euler('ZXZ',euler_rad.reshape(-1,3),degrees=False).as_quat(canonical=True).reshape(*shape,4).astype('float32')

def _sym_for_spacegroup(sg):
    if sg in (194,191,192,193):return Rotation.create_group('D6').as_quat()
    if sg in (225,229,221,227):return Rotation.create_group('O').as_quat()
    raise ValueError(f'Unimplemented space group {sg}: must add validated crystallographic symmetry')

def _sym_scalar_best(qa,qb,sym):
    # q_rel = inv(a)*b; quaternion vector and scalar products
    a=qa.astype('float64'); b=qb.astype('float64')
    va=a[:,:3]; wa=a[:,3]; vb=b[:,:3]; wb=b[:,3]
    rel_v=wa[:,None]*vb-wb[:,None]*va-np.cross(va,vb)
    rel_w=wa*wb+(va*vb).sum(axis=1)
    # max scalar of q_rel*s under right symmetry; ±quaternion equivalence
    scal=rel_w[:,None]*sym[None,:,3]-rel_v@sym[:,:3].T
    return np.max(np.abs(scal),axis=1)

def misorientation_deg(a,b,phase_a,phase_b,phase_specs,block=24000):
    """Same-phase disorientation; NaN on unindexed/interphase pairs."""
    assert a.shape==b.shape and a.shape[-1]==4
    s=a.shape[:-1]
    A=a.reshape(-1,4);B=b.reshape(-1,4)
    pa=phase_a.reshape(-1);pb=phase_b.reshape(-1)
    out=np.full(len(pa),np.nan,np.float32)
    for pid,spec in phase_specs.items():
        mask=np.flatnonzero((pa==pid)&(pb==pid))
        if not mask.size:continue
        sym=_sym_for_spacegroup(spec['space_group'])
        for start in range(0,len(mask),block):
            check_cancelled()
            ix=mask[start:start+block]
            ma=A[ix].astype(np.float64);mb=B[ix].astype(np.float64)
            ma/=np.maximum(np.linalg.norm(ma,axis=1,keepdims=True),1e-12)
            mb/=np.maximum(np.linalg.norm(mb,axis=1,keepdims=True),1e-12)
            scalar=_sym_scalar_best(ma,mb,sym)
            out[ix]=np.degrees(2*np.arccos(np.clip(scalar,0,1))).astype(np.float32)
    return out.reshape(s)

def ipf_colors(euler_rad,phase,phase_specs,axis='Z'):
    """Symmetry-aware IPF (TSL convention from orix). Not AZtec colour-calibrated."""
    try:
        from orix.quaternion import Orientation
        from orix.quaternion.symmetry import D6h, Oh
        from orix.plot import IPFColorKeyTSL
        from orix.vector import Vector3d
    except ImportError as e:
        raise RuntimeError('IPF rendering requires orix: py -m pip install orix') from e
    axes={'X':(1,0,0),'Y':(0,1,0),'Z':(0,0,1)}
    if axis.upper() not in axes:raise ValueError(axis)
    out=np.zeros((*phase.shape,3),dtype='uint8')
    pf=phase.ravel(); ef=euler_rad.reshape(-1,3); of=out.reshape(-1,3)
    for pid,spec in phase_specs.items():
        ids=np.flatnonzero(pf==pid)
        if not len(ids):continue
        sg=spec['space_group']
        sym=D6h if sg in (194,191,192,193) else Oh if sg in (225,229,221,227) else None
        if sym is None:raise ValueError('Unsupported symmetry for IPF: '+str(sg))
        key=IPFColorKeyTSL(sym,direction=Vector3d(axes[axis.upper()]))
        for k in range(0,len(ids),120000):
            check_cancelled()
            idx=ids[k:k+120000]
            ori=Orientation.from_euler(ef[idx],symmetry=sym)
            of[idx]=np.rint(255*np.clip(key.orientation2color(ori),0,1)).astype('uint8')
    return out

@transactional_output('path')
def ipf_key_images(path):
    from pathlib import Path
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    from orix.quaternion.symmetry import D6h,Oh
    from orix.plot import IPFColorKeyTSL
    from matplotlib import pyplot as plt
    for name,sym in [('HCP_alpha',D6h),('BCC_beta',Oh)]:
        key=IPFColorKeyTSL(sym)
        fig=key.plot(return_figure=True) if False else None
        # Generic orix key plot via projection; differs by orix releases
        fig=plt.figure(figsize=(3.5,3.5),dpi=160)
        ax=fig.add_subplot(111,projection='ipf',symmetry=sym)
        ax.plot_ipf_color_key()
        fig.savefig(str(path/f'IPF_KEY_{name}.png'),bbox_inches='tight')
        plt.close(fig)
