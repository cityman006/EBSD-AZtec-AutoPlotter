import json
from pathlib import Path
import shutil
import h5py
import pytest
from ebsd_autoplot import pipeline, storage
from ebsd_autoplot.archive import create_context, _safe_stem
from ebsd_autoplot.cli import main, parser
from ebsd_autoplot.compute import prepare
from ebsd_autoplot.crystal import ipf_key_images
from ebsd_autoplot.discovery import discover_scans
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.mtex import export_roi_ctf
from ebsd_autoplot.render import maps
from ebsd_autoplot.texture import texture_preview
from ebsd_autoplot.v2 import run_v2
from test_pipeline import write_synth

def source(tmp_path):
    write_synth(tmp_path/'raw')
    return H5Source(tmp_path/'raw'/'sample.h5oina')

def test_all_science_output_producers_commit_from_private_temps(tmp_path,monkeypatch):
    src=source(tmp_path);commits=[];original=storage._publish
    def observe(temp,target):
        assert '.staging' in temp.parts
        assert not target.exists()
        commits.append(target)
        original(temp,target)
    monkeypatch.setattr(storage,'_publish',observe)
    root=tmp_path/'outputs';cache=prepare(src,root/'cache',do_ipf=True)
    maps(cache,root/'render',formats=('png','tiff','pdf','svg'))
    run_v2(cache,root/'v2',roi=[0,0,16,12],min_size=1)
    texture_preview(cache,root/'texture')
    export_roi_ctf(src,root/'mtex'/'verification_roi.ctf',2,1,4,3)
    ipf_key_images(root/'ipf_keys')
    assert len(commits)>30
    assert all(storage.long_path(root) in p.parents for p in commits)
    for folder,op in [('cache','compute.prepare'),('render','render.maps'),('v2','v2.run_v2'),
                      ('texture','texture.texture_preview'),('mtex','mtex.export_roi_ctf'),('ipf_keys','crystal.ipf_key_images')]:
        assert storage.validate_receipt(root/folder,'ebsd_autoplot.'+op)
    svg=(root/'render'/'combined'/'EBSD_paper_layout.svg').read_text(encoding='utf-8')
    assert '<text' in svg and 'data:image/png;base64,' in svg
    assert 'file://' not in svg and str(tmp_path) not in svg
    ctf=(root/'mtex'/'verification_roi.ctf').read_text()
    assert 'XCells\t4' in ctf and 'YCells\t3' in ctf

def test_old_complete_cache_can_render_without_raw_or_cache_rewrite(tmp_path):
    src=source(tmp_path);context=create_context(src,tmp_path/'out')
    cache=prepare(src,context.cache_dir,do_ipf=False)
    # Simulate the shipped pre-transaction cache format, preserving its bytes.
    shutil.move(str(context.cache_dir/'.transactions'),str(tmp_path/'saved_new_receipts'))
    storage.atomic_json(context.analysis_dir/'analysis_manifest.json',{
        'schema_version':1,'dataset_id':context.dataset_id,'scan_id':src.scan_path,
        'analysis_run_id':context.analysis_run_id,'status':'success'})
    before=storage.digest(cache);src.path.rename(src.path.with_suffix('.offline'))
    assert main(['--cache',str(cache),'--font','Arial','--formats','png'])==0
    assert storage.digest(cache)==before

def test_cancel_render_has_no_partial_final_images_and_resume_keeps_cache(tmp_path,monkeypatch):
    src=source(tmp_path);opts=vars(parser().parse_args(['--no-ipf','--no-render']))
    run=pipeline.analyze(src,tmp_path/'out',opts);cache=run/'analysis_cache'/'features.h5';before=storage.digest(cache)
    real=pipeline.maps
    def cancelled(*args,**kwargs):
        kwargs['progress']=lambda _:(_ for _ in ()).throw(storage.Cancelled('cancel in renderer'))
        return real(*args,**kwargs)
    monkeypatch.setattr(pipeline,'maps',cancelled)
    with pytest.raises(storage.Cancelled):pipeline.render_only(cache,opts)
    mp=next((run.parent.parent/'renders').glob('*/render_manifest.json'))
    assert json.loads(mp.read_text())['status']=='cancelled'
    assert not list(mp.parent.glob('single_maps/*.png'))
    assert not json.loads((run.parent.parent/'latest.json').read_text()).get('render')
    monkeypatch.setattr(pipeline,'maps',real)
    pipeline.resume(mp)
    assert json.loads(mp.read_text())['status']=='success'
    assert storage.digest(cache)==before

def test_corrupt_cache_rejects_resume_without_recompute(tmp_path,monkeypatch):
    src=source(tmp_path);opts=vars(parser().parse_args(['--no-ipf','--no-render']))
    flag=[False];real=pipeline.event
    def pause(kind,path,**kw):
        real(kind,path,**kw)
        if kw.get('stage')=='v1':flag[0]=True
    monkeypatch.setattr(pipeline,'event',pause)
    with storage.cancellation(lambda:flag[0]):
        with pytest.raises(storage.Cancelled):pipeline.analyze(src,tmp_path/'out',opts)
    mp=next((tmp_path/'out').rglob('analysis_manifest.json'));cache=mp.parent/'analysis_cache'/'features.h5'
    with h5py.File(cache,'a') as f:f['features'][0,1,1]=99
    before=storage.digest(cache)
    with pytest.raises(ValueError,match='Checkpoint changed'):pipeline.resume(mp)
    assert storage.digest(cache)==before

def test_source_modified_during_analysis_is_not_successful(tmp_path,monkeypatch):
    src=source(tmp_path);opts=vars(parser().parse_args(['--no-ipf','--no-render']));real=pipeline.prepare
    def change(*args,**kw):
        result=real(*args,**kw)
        with h5py.File(src.path,'a') as f:f.attrs['changed_during_analysis']='yes'
        return result
    monkeypatch.setattr(pipeline,'prepare',change)
    with pytest.raises(ValueError,match='Input changed'):pipeline.analyze(src,tmp_path/'out',opts)
    mp=next((tmp_path/'out').rglob('analysis_manifest.json'))
    assert json.loads(mp.read_text())['status']=='failed'
    assert not list((tmp_path/'out').rglob('latest.json'))

def test_scan_added_at_same_source_path_keeps_persistent_dataset(tmp_path):
    src=source(tmp_path);first=create_context(src,tmp_path/'out')
    with h5py.File(src.path,'a') as f:f.copy('1','2')
    contexts=[create_context(H5Source(s),tmp_path/'out') for s in discover_scans(src.path)]
    assert all(c.dataset_dir==first.dataset_dir and c.dataset_id==first.dataset_id for c in contexts)
    assert len({c.scan_id for c in contexts})==2

def test_long_unicode_filename_collisions_and_reserved_names(tmp_path):
    src=source(tmp_path);contexts=[]
    for tail in ('A','B'):
        name=('中文 空格_'+('a'*140)+tail+'.h5oina')
        dest=storage.long_path(src.path.parent/name)
        shutil.copy2(src.path,dest)
        contexts.append(create_context(H5Source(dest),tmp_path/'results'))
    assert contexts[0].dataset_dir!=contexts[1].dataset_dir
    assert all(len(c.dataset_dir.name)<=72 for c in contexts)
    assert _safe_stem('CON.h5')=='_CON'
    assert _safe_stem('x<>.h5')=='x__'

def test_untrusted_transaction_cannot_publish_outside_run(tmp_path):
    root=tmp_path/'run';root.mkdir();outside=tmp_path/'outside.csv';outside.write_text('keep')
    plan=root/'plan.json';storage.atomic_json(plan,{'status':'prepared','files':[
        {'target':'../outside.csv','temp':'x','sha256':storage.digest(outside)}]})
    with pytest.raises(ValueError,match='escapes task'):storage.finish_plan(root,plan)
    assert outside.read_text()=='keep'

@pytest.mark.parametrize('boundary',['log','latest'])
def test_final_metadata_fault_never_sets_false_success_or_latest(tmp_path,monkeypatch,boundary):
    src=source(tmp_path);options=vars(parser().parse_args(['--no-ipf','--no-render']))
    attribute='_publish' if boundary=='log' else 'update_latest';real=getattr(pipeline,attribute)
    def denied(*args,**kwargs):raise PermissionError('injected final '+boundary+' write failure')
    monkeypatch.setattr(pipeline,attribute,denied)
    with pytest.raises(PermissionError):pipeline.analyze(src,tmp_path/'out',options)
    mp=next((tmp_path/'out').rglob('analysis_manifest.json'))
    assert json.loads(mp.read_text())['status']=='partial'
    assert not list((tmp_path/'out').rglob('latest.json'))
    cache=mp.parent/'analysis_cache'/'features.h5';before=storage.digest(cache)
    monkeypatch.setattr(pipeline,attribute,real)
    pipeline.resume(mp)
    assert json.loads(mp.read_text())['status']=='success' and storage.digest(cache)==before

def test_existing_suffix_folder_for_another_identity_is_never_reused(tmp_path):
    from ebsd_autoplot.archive import _source_info
    a=source(tmp_path/'a');b=source(tmp_path/'b');root=tmp_path/'out'
    first=create_context(a,root)
    ident=_source_info(b)['dataset_id'];conflict=root/('sample__src_'+ident[:8]);conflict.mkdir()
    storage.atomic_json(conflict/'dataset_manifest.json',{'dataset_id':'different-user-data'})
    sentinel=conflict/'keep.txt';sentinel.write_text('keep')
    second=create_context(b,root)
    assert storage.canonical_path(second.dataset_dir)!=conflict
    assert second.dataset_dir!=first.dataset_dir and sentinel.read_text()=='keep'
