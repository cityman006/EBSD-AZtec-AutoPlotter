"""Actual dependency, HDF5, HCP/BCC orix colour, and PNG round-trip checks."""
from datetime import datetime
import importlib
import importlib.metadata
import json
from pathlib import Path
import struct
import sys

def main():
    assert sys.version_info[:2] == (3, 12), sys.version
    assert struct.calcsize('P') == 8
    root = Path(__file__).resolve().parent
    assert Path(sys.prefix).resolve() == root / '.venv', sys.executable
    target = root / 'output' / ('verification_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    target.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name, package in [('numpy','numpy'),('scipy','scipy'),('h5py','h5py'),
                          ('matplotlib','matplotlib'),('PIL','Pillow'),('orix','orix'),
                          ('numba','numba'),('CifFile','PyCifRW'),('tkinter',None),('ebsd_autoplot',None)]:
        module = importlib.import_module(name)
        versions[name] = importlib.metadata.version(package) if package else str(getattr(module,'TkVersion','project'))
    import numpy as np
    import h5py
    from PIL import Image
    from ebsd_autoplot.crystal import ipf_colors
    from orix.quaternion import Orientation
    from orix.quaternion.symmetry import D6h, Oh
    from orix.vector import Vector3d
    from orix.plot import IPFColorKeyTSL
    data = np.arange(120,dtype=np.float32).reshape(10,12)
    with h5py.File(target/'roundtrip.h5','w') as f:
        f.create_dataset('test',data=data,compression='gzip')
    with h5py.File(target/'roundtrip.h5','r') as f:
        np.testing.assert_array_equal(f['test'][()],data)
    y,x = np.mgrid[0:16,0:20]
    euler = np.stack([x/20*np.pi*2, y/16*np.pi/2, (x+y)/36*np.pi],axis=-1)
    phase = np.ones((16,20),dtype=np.uint8)
    for name,sg,sym in [('HCP',194,D6h),('BCC',229,Oh)]:
        for axis,direction in zip('XYZ',[(1,0,0),(0,1,0),(0,0,1)]):
            rgb = ipf_colors(euler,phase,{1:{'space_group':sg}},axis=axis)
            assert rgb.shape==(16,20,3) and rgb.dtype==np.uint8
            assert len(np.unique(rgb.reshape(-1,3),axis=0))>8
            # Compare the public project function against direct orix output.
            direct=IPFColorKeyTSL(sym,direction=Vector3d(direction)).orientation2color(
                Orientation.from_euler(euler.reshape(-1,3),symmetry=sym))
            assert np.isfinite(direct).all() and (direct>=0).all() and (direct<=1).all()
            np.testing.assert_array_equal(rgb.reshape(-1,3),np.rint(direct*255).astype('u1'))
            path=target/f'{name}_IPF_{axis}.png'
            Image.fromarray(rgb).save(path)
            with Image.open(path) as im:
                im.load()
                assert im.mode=='RGB'
                np.testing.assert_array_equal(np.asarray(im),rgb)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots();ax.plot([0,1,2],[0,1,4]);fig.savefig(target/'plot.png');plt.close(fig)
    with Image.open(target/'plot.png') as im:
        im.load();assert im.width>0 and im.height>0
    result={'status':'passed','python':sys.executable,'version':sys.version,'bits':64,
            'versions':versions,'output':str(target),'ipf':'orix TSL; Oxford AZtec uncalibrated',
            'checks':['imports','HDF5 compressed roundtrip','HCP and BCC XYZ IPF RGB roundtrips','matplotlib PNG']}
    (target/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
