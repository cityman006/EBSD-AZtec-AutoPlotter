import csv,json
import h5py
import numpy as np
import pytest
from ebsd_autoplot.slicing import export_patches
from ebsd_autoplot.graph_builder import build_graph
from ebsd_autoplot.compute import prepare
from ebsd_autoplot.v2 import run_v2
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.storage import digest
from ebsd_autoplot.studio_actions import run_action
from test_v4_patches import feature_cache
from test_pipeline import write_synth


def test_selected_diagonal_tiles_do_not_overstate_coverage(tmp_path):
    cache,_=feature_cache(tmp_path,shape=(6,8))
    out=tmp_path/'subset';m=export_patches(cache,out,grid=(2,2),selected_patches=[0,3],preview=False)
    assert m['patch_count']==2 and m['covered_source_pixels']==24 and m['dropped_source_pixels']==24
    assert [r['patch_id'] for r in m['patches']]==['patch_00000','patch_00003']
    with (out/'patch_manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and int(rows[0]['indexed_count'])==12


def test_new_v2_source_hash_rejects_wrong_features(tmp_path):
    write_synth(tmp_path/'raw')
    cache=prepare(H5Source(tmp_path/'raw/sample.h5oina'),tmp_path/'cache',do_ipf=False,progress=lambda _:None)
    run_v2(cache,tmp_path/'v2',min_size=1,progress=lambda _:None)
    with h5py.File(tmp_path/'v2/v2_grains.h5','r') as f:assert f.attrs['source_features_sha256']==digest(cache)
    with h5py.File(cache,'a') as f:f['features'][6,0,0]=1
    with pytest.raises(ValueError,match='different features'):
        build_graph(cache,tmp_path/'v2',tmp_path/'wrong',preview=False)


def test_archived_numeric_patch_action_does_not_reanalyze(tmp_path,monkeypatch):
    from ebsd_autoplot.pipeline import analyze
    from ebsd_autoplot.cli import parser
    write_synth(tmp_path/'raw')
    opts=vars(parser().parse_args(['--analysis','v1','--no-ipf','--no-render']))
    src=H5Source(tmp_path/'raw/sample.h5oina');run=analyze(src,tmp_path/'out',opts)
    cache=run/'analysis_cache/features.h5';before=digest(cache)
    import ebsd_autoplot.compute as compute
    def forbidden(*a,**k):raise AssertionError('Raw analysis must not run')
    monkeypatch.setattr(compute,'prepare',forbidden)
    target=run_action('patches',cache,tmp_path/'out',config={'grid':[3,5],'preview':False})
    assert json.loads((target/'results/patch_manifest.json').read_text())['patch_count']==15
    assert digest(cache)==before


def test_optional_cpu_torch_dataloader_preserves_masks_and_metadata(tmp_path):
    torch=pytest.importorskip('torch')
    import pickle
    from ebsd_autoplot.ml_dataset import make_torch_dataset,make_torch_dataloader
    cache,_=feature_cache(tmp_path,shape=(7,11))
    with h5py.File(cache,'a') as f:
        meta=json.loads(f.attrs['metadata_json']);meta['optional_transform']=None;f.attrs['metadata_json']=json.dumps(meta)
    out=tmp_path/'db';export_patches(cache,out,grid=(3,5),edge_policy='pad-with-mask',preview=False)
    index=out/'dataset_index.json';ds=make_torch_dataset(index,channels=['qx','kam_deg'])
    assert pickle.loads(pickle.dumps(ds))[0]['channels']['qx'].dtype==torch.float32
    batch=next(iter(make_torch_dataloader(index,channels=['qx','kam_deg'],batch_size=2)))
    assert batch['channels']['qx'].shape[0]==2 and batch['masks']['kam_deg'].dtype==torch.bool
    assert batch['metadata'][0]['source_metadata']['optional_transform'] is None


@pytest.mark.parametrize('pid,sg,name,expect_alpha',[(7,194,'Titanium-Hexagonal',True),(42,229,'Titanium-Cubic',False)])
def test_v2_bor_follows_phase_metadata_not_raster_one(tmp_path,pid,sg,name,expect_alpha):
    write_synth(tmp_path/'raw')
    cache=prepare(H5Source(tmp_path/'raw/sample.h5oina'),tmp_path/'cache',do_ipf=False,progress=lambda _:None)
    with h5py.File(cache,'a') as f:
        phase=f['features'][4];phase[phase>0]=pid;f['features'][4]=phase
        meta=json.loads(f.attrs['metadata_json']);meta['phase_specs']={str(pid):{'name':name,'space_group':sg}}
        f.attrs['metadata_json']=json.dumps(meta)
    result=run_v2(cache,tmp_path/'v2',min_size=1,progress=lambda _:None)
    assert bool(result['bor']['ideal_variant_pair_angles_deg']) is expect_alpha
    with (tmp_path/'v2/grain_neighbour_graph.csv').open() as f:rows=list(csv.DictReader(f))
    assert rows
    assert all(np.isfinite(float(row['alpha_alpha_misorientation_deg']))==expect_alpha for row in rows)
