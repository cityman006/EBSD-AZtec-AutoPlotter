"""Texture: orientation stereographic pole distributions and orientation histograms.

This is a projection of pole density (not a full ODF inversion). For final
ODF/pole figures compare with MTEX and validate crystal/sample conventions.
"""
from pathlib import Path
from .storage import transactional_output, check_cancelled
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


@transactional_output()
def texture_preview(features_file,outdir,max_sample=150000,seed=11):
    outdir=Path(outdir);outdir.mkdir(exist_ok=True,parents=True)
    with h5py.File(features_file,'r') as f:
        h,w=f['indexed'].shape
        n=int(h*w)
        rng=np.random.default_rng(seed)
        # sample positions evenly with jitter, avoid loading huge field
        stride=max(1,int(np.sqrt(n/max_sample)))
        ph=f['features'][4,::stride,::stride].astype(np.uint8)
        q=f['features'][0:4,::stride,::stride]
        q=np.moveaxis(q,0,-1)
    check_cancelled()
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),dpi=165,layout='constrained')
    phases=[(1,'α-Ti: basal [0001] pole',np.array([[0,0,1]])),
            (2,'β-Ti: {100} family pole',np.eye(3))]
    for ax,(pid,title,crystal_dirs) in zip(axes,phases):
        qq=q[ph==pid]
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim(-1.04,1.04);ax.set_ylim(-1.04,1.04)
        ax.add_patch(plt.Circle((0,0),1,fill=False,ec='black'))
        if len(qq):
            # budget: cap sampled points per phase
            if len(qq)>max_sample:
                qq=qq[rng.choice(len(qq),size=max_sample,replace=False)]
            # Rotation convention depends on mapping; treat as preliminary until MTEX validation
            rot=Rotation.from_quat(qq)
            v=np.concatenate([rot.apply(np.tile(d,(len(qq),1))) for d in crystal_dirs],axis=0)
            v[v[:,2]<0]*=-1
            den=1+v[:,2]
            xx=v[:,0]/den;yy=v[:,1]/den
            counts,xe,ye=np.histogram2d(xx,yy,bins=120,range=[[-1,1],[-1,1]])
            ax.imshow(np.log1p(counts).T,extent=(-1,1,-1,1),origin='lower',cmap='inferno')
            ax.add_patch(plt.Circle((0,0),1,fill=False,ec='white',lw=1.5))
        ax.set_xlabel('X stereographic');ax.set_ylabel('Y stereographic')
    fig.suptitle('Exploratory pole distributions | NOT full ODF, AZtec/MTEX calibration pending',fontsize=11)
    path=outdir/'texture_pole_preview.png';fig.savefig(path);plt.close(fig)
    return path
