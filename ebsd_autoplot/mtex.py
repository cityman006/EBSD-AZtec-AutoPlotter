"""Export a selected ROI as Oxford CTF for MTEX cross-validation."""
from pathlib import Path
import numpy as np
from .storage import transactional_output, check_cancelled
from .v3_crystal import ORIENTATION_CONVENTION

@transactional_output('path', file_target=True)
def export_roi_ctf(source,path,x,y,w,h):
    if max(w,h)>5000 or w*h>3000000:
        raise ValueError('Export an ROI no larger than 3 million points for MTEX verification')
    if min(x,y)<0 or min(w,h)<1 or x+w>source.nx or y+h>source.ny:raise ValueError('ROI outside scan')
    d=source.read(y,y+h,('phase','euler'))
    d={k:v[:,x:x+w] for k,v in d.items()}
    phase=d['phase'];e=d['euler']
    source_unit=source.probe_euler_unit()
    if source_unit=='radian':e=np.rad2deg(e)
    sp=source.phase_specs
    def cell(spec):
        lattice=spec.get('lattice_parameters')
        if not lattice:
            raise ValueError(f'Measured lattice parameters are missing for phase {spec["name"]}; refusing to write nominal constants')
        if int(spec['space_group']) not in (194,229):
            raise ValueError(f'V3 MTEX export only validates Ti HCP/BCC phase symmetries, got {spec["space_group"]}')
        dims=';'.join(f'{float(lattice[k]):.8g}' for k in ('a','b','c'))
        angles=';'.join(f'{float(lattice[k+"_deg"]):.8g}' for k in ('alpha','beta','gamma'))
        return dims+'\t'+angles+'\t'+spec['name']+'\t11\t'+str(spec['space_group'])+'\t0'
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
    import json
    path.with_suffix('.orientation.json').write_text(json.dumps({
        'euler_input_unit':source_unit,'ctf_euler_unit':'degree',
        'orientation_convention':ORIENTATION_CONVENTION,
        'phase_id_mapping':{str(k):{'name':v['name'],'space_group':v['space_group'],
                                    'lattice_parameters':v.get('lattice_parameters')} for k,v in sp.items()},
        'specimen_orientation_euler':source.header.get('specimen_orientation_euler'),
        'scanning_rotation_rad':source.header.get('scanning_rotation_rad'),
        'rotation_policy':'preserved as source metadata; not applied to Euler values pending instrument-frame validation',
        'roi_xywh':[int(x),int(y),int(w),int(h)],
        'scan_path':source.scan_path,
    },ensure_ascii=False,indent=2),encoding='utf-8')
    return path
