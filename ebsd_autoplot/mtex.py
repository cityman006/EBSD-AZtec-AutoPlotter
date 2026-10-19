"""Export a selected ROI as Oxford CTF for MTEX cross-validation."""
from pathlib import Path
import numpy as np
from .storage import transactional_output, check_cancelled

@transactional_output('path', file_target=True)
def export_roi_ctf(source,path,x,y,w,h):
    if max(w,h)>5000 or w*h>3000000:
        raise ValueError('Export an ROI no larger than 3 million points for MTEX verification')
    if min(x,y)<0 or min(w,h)<1 or x+w>source.nx or y+h>source.ny:raise ValueError('ROI outside scan')
    d=source.read(y,y+h,('phase','euler'))
    d={k:v[:,x:x+w] for k,v in d.items()}
    phase=d['phase'];e=d['euler']
    if source.probe_euler_unit()=='radian':e=np.rad2deg(e)
    sp=source.phase_specs
    def cell(spec):
        if spec['space_group'] in (194,191):return '2.951;2.951;4.684\t90;90;120\t'+spec['name']+'\t11\t194\t0'
        if spec['space_group'] in (229,225):return '3.311;3.311;3.311\t90;90;90\t'+spec['name']+'\t11\t'+str(spec['space_group'])+'\t0'
        raise ValueError('Manual lattice dimensions needed for '+spec['name'])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='ascii',newline='\n') as f:
        f.write('Channel Text File\nPrj\tEBSD AutoPlotter MTEX verification\nAuthor\tEBSD-AutoPlotter\n')
        f.write('JobMode\tGrid\nXCells\t%d\nYCells\t%d\nXStep\t%.8f\nYStep\t%.8f\nAcqE1\t0\nAcqE2\t0\nAcqE3\t0\n' %(w,h,source.sx,source.sy))
        f.write('Euler angles in degrees\nPhases\t%d\n'%len(sp))
        for k in sorted(sp):f.write(cell(sp[k])+'\n')
        f.write('Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n')
        for iy in range(h):
            check_cancelled()
            for ix in range(w):
                a,b,c=e[iy,ix]
                f.write(f'{int(phase[iy,ix])}\t{ix*source.sx:.5f}\t{iy*source.sy:.5f}\t0\t0\t{a:.6f}\t{b:.6f}\t{c:.6f}\t0\t0\t0\n')
    return path
