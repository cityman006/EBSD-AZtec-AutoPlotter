import errno
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest

from ebsd_autoplot import pipeline, storage
from ebsd_autoplot.archive import create_context
from ebsd_autoplot.cli import main, parser
from ebsd_autoplot.discovery import discover_scans
from ebsd_autoplot.io import H5Source
from test_pipeline import write_synth

def opts(*extra):
    return vars(parser().parse_args(['--no-ipf','--no-render',*extra]))

def source(tmp_path):
    write_synth(tmp_path/'raw')
    return H5Source(tmp_path/'raw'/'sample.h5oina')

def manifest(root):
    mp=next(root.rglob('analysis_manifest.json'))
    return mp,json.loads(mp.read_text(encoding='utf-8'))

def test_multi_scan_real_ipf_arrays_and_output_scopes(tmp_path):
    src=source(tmp_path);raw=src.path
    with h5py.File(raw,'a') as f:
        f.copy('1','2')
        for key in ('1','2'):
            ph=f[key+'/EBSD/Data/Phase'];p=ph[()];p[50:]=2;ph[:]=p
        f['2/EBSD/Data/Band Contrast'][:]=200
        f['2/EBSD/Data/Euler'][:,0]=.9
    before=storage.digest(raw)
    root=tmp_path/'results'
    # Actual orix IPF is required here; no --no-ipf fallback.
    assert main([str(raw),'--output-root',str(root),'--analysis','v1v2','--roi','0','0','16','12'])==0
    assert storage.digest(raw)==before
    datasets=list(root.glob('*/dataset_manifest.json'));assert len(datasets)==1
    scopes=sorted(datasets[0].parent.glob('scans/scan_*'));assert len(scopes)==2
    caches=[]
    for scope in scopes:
        mp=next(scope.glob('analysis_runs/*/analysis_manifest.json'))
        assert json.loads(mp.read_text())['status']=='success'
        cache=mp.parent/'analysis_cache'/'features.h5';caches.append(cache)
        with h5py.File(cache,'r') as f:
            assert set(('ipf_X','ipf_Y','ipf_Z'))<=set(f)
            assert f['ipf_X'].shape==(12,16,3)
        assert list(scope.glob('renders/*/combined/EBSD_paper_layout.png'))
        assert list(scope.glob('renders/*/legends/MAD_colourbar.png'))
        assert (scope/'latest.json').is_file()
    with h5py.File(caches[0],'r') as a,h5py.File(caches[1],'r') as b:
        assert not np.array_equal(a['ipf_X'][()],b['ipf_X'][()])
        assert np.all(b['features'][6]==200)

def montage(tmp_path,*,different_names=False):
    root=tmp_path/'montage';root.mkdir();scans=[]
    for index in (1,2):
        scratch=tmp_path/f'parts{index}';write_synth(scratch,split=True)
        name=f'原始 材料{index}.h5oina' if different_names else '原始 材料.h5oina'
        entries=[]
        for old in sorted(scratch.glob('*.h5')):
            new=root/old.name.replace('scan01',f'scan{index:02d}')
            shutil.copy2(old,new)
            with h5py.File(new,'a') as f:
                if index==2:f.copy('1','2');del f['1']
                f.attrs['source_name']=name;f.attrs['scan_id']=f'scan{index:02d}'
            start=int(new.name.split('rows')[1].split('-')[0]);stop=start+6
            entries.append({'file':new.name,'row_start':start,'row_end_exclusive':stop})
        scans.append({'scan_path':f'/{index}/EBSD','scan_id':f'scan{index:02d}',
                      'source_name':name,'parts':list(reversed(entries))})
    mp=root/'请上传_分析清单.json';mp.write_text(json.dumps({'scans':list(reversed(scans))},ensure_ascii=False),encoding='utf-8')
    return root,mp

@pytest.mark.parametrize('different_names',[False,True])
def test_montage_verified_names_groups_and_reordered_parts(tmp_path,different_names):
    root,mp=montage(tmp_path,different_names=different_names)
    specs=discover_scans(root);assert len(specs)==2
    contexts=[create_context(H5Source(s),tmp_path/'results') for s in specs]
    assert len({c.dataset_dir for c in contexts})==(2 if different_names else 1)
    assert all(c.dataset_dir.name.startswith('原始 材料') for c in contexts)
    assert all('scan01_rows' not in c.dataset_dir.name for c in contexts)
    old_ids=[c.dataset_id for c in contexts]
    data=json.loads(mp.read_text(encoding='utf-8'));data['scans'].reverse()
    for scan in data['scans']:scan['parts'].reverse()
    mp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    assert [create_context(H5Source(s),tmp_path/'results').dataset_id for s in discover_scans(root)]==old_ids

@pytest.mark.parametrize('malice',['../escape.h5oina','C:\\escape.h5oina','/absolute.h5'])
def test_manifest_source_name_cannot_escape(tmp_path,malice):
    root,mp=montage(tmp_path)
    data=json.loads(mp.read_text(encoding='utf-8'));data['scans'][0]['source_name']=malice
    mp.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='Unsafe source_name'):discover_scans(root)

@pytest.mark.parametrize('fault',['name','range','scan','width','length','path'])
def test_bad_montage_fails_before_archiving(tmp_path,fault):
    root,mp=montage(tmp_path);data=json.loads(mp.read_text(encoding='utf-8'));sc=data['scans'][0];part=sc['parts'][0]
    if fault=='name':sc['source_name']='wrong.h5oina'
    elif fault=='range':part['row_start']+=1
    elif fault=='path':part['file']='../outside.h5'
    else:
        with h5py.File(root/part['file'],'a') as f:
            if fault=='scan':f.attrs['scan_id']='other'
            elif fault=='width':f[sc['scan_path']+'/Header/X Cells'][0]=15
            else:
                key=sc['scan_path']+'/Data/Phase';del f[key];f[key]=np.ones(2,dtype='u1')
    mp.write_text(json.dumps(data),encoding='utf-8')
    output=tmp_path/'results'
    assert main([str(root),'--output-root',str(output),'--no-render','--no-ipf'])==1
    assert not list(output.rglob('analysis_manifest.json'))

def test_cancel_after_v1_resume_without_raw_or_recompute(tmp_path,monkeypatch):
    src=source(tmp_path);root=tmp_path/'results';requested=[False]
    original=pipeline.event
    def checkpoint(kind,path,**kwargs):
        original(kind,path,**kwargs)
        if kwargs.get('stage')=='v1':requested[0]=True
    monkeypatch.setattr(pipeline,'event',checkpoint)
    with storage.cancellation(lambda:requested[0]):
        with pytest.raises(storage.Cancelled):pipeline.analyze(src,root,opts())
    mp,m=manifest(root);assert m['status']=='cancelled'
    cache=mp.parent/'analysis_cache'/'features.h5';before=storage.digest(cache)
    src.path.rename(src.path.with_suffix('.unavailable'))  # Only the temporary fixture is renamed.
    def forbidden(*args,**kwargs):raise AssertionError('V1 must not run again')
    monkeypatch.setattr(pipeline,'prepare',forbidden);monkeypatch.setattr(pipeline,'_source_for',forbidden)
    monkeypatch.setattr(pipeline,'event',original)
    assert pipeline.resume(mp)==storage.long_path(mp.parent)
    assert json.loads(mp.read_text())['status']=='success'
    assert storage.digest(cache)==before

def test_cancel_during_compute_has_no_final_hdf5_and_can_restart_stage(tmp_path,monkeypatch):
    src=source(tmp_path);root=tmp_path/'results';original=pipeline.prepare
    def interrupted(*args,**kwargs):
        def progress(message):raise storage.Cancelled('test cancellation during first block')
        return original(*args,**kwargs,progress=progress)
    monkeypatch.setattr(pipeline,'prepare',interrupted)
    with pytest.raises(storage.Cancelled):pipeline.analyze(src,root,opts())
    mp,m=manifest(root);assert m['status']=='cancelled'
    assert not (mp.parent/'analysis_cache'/'features.h5').exists()
    assert list(mp.parent.rglob('.staging/*'))
    monkeypatch.setattr(pipeline,'prepare',original)
    pipeline.resume(mp)
    assert json.loads(mp.read_text())['status']=='success'

def test_v2_failure_retains_v1_and_resumes_from_verified_cache(tmp_path,monkeypatch):
    src=source(tmp_path);root=tmp_path/'results';original=pipeline.run_v2
    def fail(*a,**kw):raise RuntimeError('injected V2 failure')
    monkeypatch.setattr(pipeline,'run_v2',fail)
    with pytest.raises(RuntimeError):pipeline.analyze(src,root,opts('--analysis','v1v2','--roi','0','0','16','12'))
    mp,m=manifest(root);assert m['status']=='partial' and 'v1' in m['checkpoints']
    cache=mp.parent/'analysis_cache'/'features.h5';before=storage.digest(cache)
    monkeypatch.setattr(pipeline,'run_v2',original)
    monkeypatch.setattr(pipeline,'prepare',lambda *a,**kw:pytest.fail('V1 was repeated'))
    pipeline.resume(mp);assert storage.digest(cache)==before
    assert (mp.parent/'analysis_cache'/'v2'/'v2_grains.h5').exists()

def test_changed_source_and_tampered_cache_are_not_resumed(tmp_path,monkeypatch):
    src=source(tmp_path);root=tmp_path/'results';original=pipeline.prepare
    monkeypatch.setattr(pipeline,'prepare',lambda *a,**kw:(_ for _ in ()).throw(storage.Cancelled('pause')))
    with pytest.raises(storage.Cancelled):pipeline.analyze(src,root,opts())
    mp,_=manifest(root)
    with h5py.File(src.path,'a') as f:f.attrs['revision']='new'
    monkeypatch.setattr(pipeline,'prepare',original)
    with pytest.raises(ValueError,match='version changed'):pipeline.resume(mp)
    assert not (mp.parent/'analysis_cache'/'features.h5').exists()

@pytest.mark.parametrize('failure',[errno.ENOSPC,errno.EACCES])
def test_atomic_writer_faults_expose_no_partial_finals(tmp_path,failure):
    out=tmp_path/'out'
    @storage.transactional_output()
    def writer(outdir):
        (outdir/'table.csv').write_text('partial')
        raise OSError(failure,'injected filesystem error')
    with pytest.raises(OSError):writer(out)
    assert not (out/'table.csv').exists()

def test_interrupted_commit_and_tampered_plan_recovery(tmp_path,monkeypatch):
    out=tmp_path/'out';original=storage._publish;count=[0]
    @storage.transactional_output()
    def writer(outdir):
        (outdir/'a.csv').write_text('complete A');(outdir/'b.csv').write_text('complete B')
    def interrupt(temp,target):
        if count[0]:raise OSError('injected mid-commit failure')
        original(temp,target);count[0]+=1
    monkeypatch.setattr(storage,'_publish',interrupt)
    with pytest.raises(OSError):writer(out)
    assert (out/'a.csv').read_text()=='complete A' and not (out/'b.csv').exists()
    monkeypatch.setattr(storage,'_publish',original)
    storage.recover_transactions(out)
    assert (out/'b.csv').read_text()=='complete B'
    assert storage.validate_receipt(out,__name__+'.writer')
    (out/'a.csv').write_text('changed')
    with pytest.raises(ValueError,match='Checkpoint changed'):storage.validate_receipt(out,__name__+'.writer')

def test_no_overwrite_of_prior_complete_artifact(tmp_path):
    out=tmp_path/'out';out.mkdir();old=out/'IPF_X.png';old.write_bytes(b'old')
    @storage.transactional_output()
    def writer(outdir):(outdir/'IPF_X.png').write_bytes(b'new')
    with pytest.raises(FileExistsError):writer(out)
    assert old.read_bytes()==b'old'

def test_hard_interruption_owner_recovery_and_resume(tmp_path):
    src=source(tmp_path);root=tmp_path/'results';ready=tmp_path/'ready.json'
    script="""
import json,sys,time
from pathlib import Path
from ebsd_autoplot import pipeline
from ebsd_autoplot.cli import main
original=pipeline.event
def event(kind,path,**kwargs):
    original(kind,path,**kwargs)
    if kwargs.get('stage')=='v1':
        Path(sys.argv[3]).write_text(json.dumps({'run':str(path)}))
        while True:time.sleep(.05)
pipeline.event=event
raise SystemExit(main([sys.argv[1],'--output-root',sys.argv[2],'--no-ipf','--no-render']))
"""
    proc=subprocess.Popen([sys.executable,'-u','-c',script,str(src.path),str(root),str(ready)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+25
        while not ready.exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.05)
        assert ready.exists(),'worker did not reach the real V1 checkpoint'
        mp,m=manifest(root);cache=mp.parent/'analysis_cache'/'features.h5';before=storage.digest(cache)
        assert pipeline.recover_interrupted(root)==[]  # A live owner is never marked abandoned.
        with pytest.raises(ValueError,match='still running'):pipeline.resume(mp)
        proc.kill();proc.wait(timeout=10)
        assert pipeline.recover_interrupted(root)==[str(mp)]
        assert json.loads(mp.read_text())['status']=='partial'
        pipeline.resume(mp);assert storage.digest(cache)==before
        assert json.loads(mp.read_text())['status']=='success'
    finally:
        if proc.poll() is None:proc.kill();proc.wait(timeout=10)

def test_real_process_kill_during_atomic_commit_finishes_verified_plan(tmp_path):
    root=tmp_path/'artifacts';ready=tmp_path/'ready_commit.json'
    script="""
import sys,time
from pathlib import Path
from ebsd_autoplot import storage
publish=storage._publish
def pause(temp,target):
    publish(temp,target)
    Path(sys.argv[2]).write_text('ready')
    while True:time.sleep(.05)
storage._publish=pause
@storage.transactional_output()
def writer(outdir):
    (outdir/'a.csv').write_text('complete a')
    (outdir/'b.csv').write_text('complete b')
writer(Path(sys.argv[1]))
"""
    proc=subprocess.Popen([sys.executable,'-c',script,str(root),str(ready)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+15
        while not ready.exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.03)
        assert ready.exists()
        proc.kill();proc.wait(timeout=10)
        assert (root/'a.csv').read_text()=='complete a' and not (root/'b.csv').exists()
        storage.recover_transactions(root)
        assert (root/'b.csv').read_text()=='complete b'
        plan=json.loads(next((root/'.transactions').glob('*.json')).read_text())
        assert plan['status']=='committed'
    finally:
        if proc.poll() is None:proc.kill();proc.wait(timeout=10)
