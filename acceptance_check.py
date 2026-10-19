"""Read and validate a fresh real ROI result; never edit raw EBSD data."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from PIL import Image
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.storage import atomic_json,long_path,canonical_path
from datetime import datetime

source=H5Source(sys.argv[1])
root=long_path(sys.argv[2]);out=root;images=root;report_dir=root
if not (root/'features.h5').exists():
    runs=[]
    for mp in root.rglob('analysis_manifest.json'):
        m=json.loads(mp.read_text(encoding='utf-8'))
        if m.get('status')=='success' and m.get('input_path')==str(source.path) and m.get('kind')=='analysis':runs.append((m.get('completed_at',''),mp,m))
    assert runs,'No successful archived analysis matches the selected source'
    _,mp,m=max(runs,key=lambda row:row[0]);out=mp.parent/'analysis_cache';report_dir=mp.parent/'validation'
    dataset=next(p for p in mp.parents if (p/'dataset_manifest.json').exists())
    images=dataset/m['render_path']
expected=(source.ny,source.nx)
data=source.read(0,source.ny,('phase','bc','pq','mad'))
with h5py.File(out/'features.h5','r') as f:
    arrays={key:f[key][()] for key in f}
    features=arrays['features']
    assert features.shape==(7,*expected)
    assert features.dtype==np.float32
    indexed=arrays['indexed'].astype(bool)
    valid=arrays['kam_valid'].astype(bool)
    np.testing.assert_array_equal(indexed,data['phase']>0)
    np.testing.assert_array_equal(features[4],data['phase'])
    np.testing.assert_array_equal(features[6],data['bc'])
    assert indexed.any() and valid.any()
    assert not (valid & ~indexed).any()
    assert np.isfinite(features[:,indexed]).all()
    np.testing.assert_allclose(np.linalg.norm(features[:4,indexed],axis=0),1,atol=1e-6)
    assert ((features[5,valid]>=0)&(features[5,valid]<=5)).all()
    assert (features[5,~valid]==0).all()
    assert (features[:4,~indexed]==0).all()
    for field in ['pq','mad']:
        if field in data:
            np.testing.assert_array_equal(arrays[field],data[field])
    for field in ['boundary_right','boundary_down']:
        assert set(np.unique(arrays[field])).issubset({0,1,2,3,255})
    for field in ['angle_right_deg','angle_down_deg']:
        a=arrays[field]
        assert not np.isinf(a).any()
        assert ((a[np.isfinite(a)]>=0)&(a[np.isfinite(a)]<=180)).all()
    for axis in 'XYZ':
        rgb=arrays['ipf_'+axis]
        assert rgb.shape==(*expected,3) and rgb.dtype==np.uint8
        assert (rgb[~indexed]==0).all()
        assert (rgb[indexed]!=0).any()
        with Image.open(images/'single_maps'/f'IPF_{axis}.png') as im:
            im.load();assert im.mode=='RGB'
            np.testing.assert_array_equal(np.asarray(im),rgb)
    channels=json.loads(f.attrs['channels'])
    assert channels==['qx','qy','qz','qw','phase_id','kam_deg','band_contrast']
required=['IPF_X','IPF_Y','IPF_Z','Phase','KAM','BC','MAD','PQ']
for name in required:
    with Image.open(images/'single_maps'/f'{name}.png') as im:
        im.load();assert im.size==(source.nx,source.ny)
pngs=[]
for path in [*images.rglob('*.png'),*out.rglob('*.png')]:
    if '.staging' in path.parts:continue
    with Image.open(path) as im:
        im.load();assert im.width>0 and im.height>0
        pngs.append({'path':str(path),'size':list(im.size),'mode':im.mode})
assert (images/'combined'/'EBSD_paper_layout.png').exists() or (images/'EBSD_paper_layout.png').exists()
assert (out/'texture'/'texture_pole_preview.png').exists() or (out/'v2'/'texture_pole_preview.png').exists()
with (out/'v2'/'grains.csv').open(encoding='utf-8',newline='') as stream:
    grains=list(csv.DictReader(stream))
summary=json.loads((out/'v2'/'v2_summary.json').read_text(encoding='utf-8'))
assert len(grains)==summary['reported_grains']>0
with h5py.File(out/'v2'/'v2_grains.h5','r') as f:
    ids=f['grain_id'][()];grod=f['GROD_deg'][()]
    assert ids.shape==grod.shape==(64,64)
    roi_indexed=indexed[:64,:64]
    np.testing.assert_array_equal(ids>0,roi_indexed)
    reported=np.isin(ids,[int(row['grain_id']) for row in grains])
    assert np.isfinite(grod[reported]).all()
    assert (grod[reported]>=0).all()
    assert np.isnan(grod[~reported]).all()
    for row in grains:
        mask=ids==int(row['grain_id'])
        assert int(row['pixels'])==int(mask.sum())
        np.testing.assert_allclose(float(row['GOS_deg']),grod[mask].mean(),atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm([float(row['mean_'+k]) for k in ['qx','qy','qz','qw']]),1,atol=1e-5)
with (out/'v2'/'grain_neighbour_graph.csv').open(encoding='utf-8',newline='') as stream:
    graph=list(csv.DictReader(stream))
    for row in graph:
        assert int(row['grain_a'])<int(row['grain_b']) and int(row['shared_pixel_edges'])>0
ctf=(out/'mtex'/'verification_roi.ctf').read_text(encoding='utf-8').splitlines()
assert ctf[0]=='Channel Text File'
header=next(i for i,line in enumerate(ctf) if line.startswith('Phase\tX\tY'))
assert len(ctf[header+1:])==4096
for path in out.rglob('*.json'):
    json.loads(path.read_text(encoding='utf-8'))
result={'status':'passed','sample_kind':'bundled real montage ROI','shape':list(expected),
        'indexed_pixels':int(indexed.sum()),'kam_valid_pixels':int(valid.sum()),
        'v2_reported_grains':len(grains),'graph_edges':len(graph),'pngs':pngs,
        'missing_values':'unindexed quaternion/IPF=0; invalid KAM=0 with kam_valid=0; edge angles=NaN when undefined; GROD=NaN for invalid or unreported small grains',
        'mtex':'CTF export read and checked; MATLAB/MTEX reconstruction not run',
        'raw_sha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()}
atomic_json(report_dir/('acceptance_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'),result)
print(json.dumps(result,indent=2))
