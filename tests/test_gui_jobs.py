"""Windows Tk + real subprocess workflows, including deterministic launch races."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from ebsd_autoplot import jobs, pipeline, storage
from ebsd_autoplot.cli import main
from ebsd_autoplot.jobs import JobManager
from test_pipeline import write_synth

PROJECT=Path(__file__).resolve().parents[1]

def raw(folder):
    write_synth(folder)
    return folder/'sample.h5oina'

def wait_jobs(manager,app=None,timeout=110):
    deadline=time.monotonic()+timeout
    while not all(h.done.is_set() for h in manager.jobs.values()) and time.monotonic()<deadline:
        if app:app.root.update()
        time.sleep(.02)
    assert all(h.done.is_set() for h in manager.jobs.values()),'job timeout'

def test_real_tk_input_output_font_switches_while_two_jobs_are_queued(tmp_path,monkeypatch):
    from gui import App
    app=App();app.root.withdraw();app.preferences=tmp_path/'prefs.json'
    a=raw(tmp_path/'A');b=raw(tmp_path/'B')
    before={str(p):storage.digest(p) for p in (a,b)}
    gate=threading.Event();original=jobs.subprocess.Popen
    def delayed(*args,**kwargs):
        if not gate.wait(10):raise RuntimeError('launch gate timed out')
        return original(*args,**kwargs)
    monkeypatch.setattr(jobs.subprocess,'Popen',delayed)
    try:
        app.infile.set(str(a));app.outfile.set(str(tmp_path/'out_A'));app.font.set('Arial');app.formats.set('png');app.an.set('v1')
        app.start();first=app.active_job.get()
        app.infile.set(str(b));app.outfile.set(str(tmp_path/'out_B'));app.font.set('Times New Roman')
        app.start();second=app.active_job.get()
        assert first!=second
        app.infile.set('Z:\\changed_while_running.h5');app.outfile.set(str(tmp_path/'out_C'));app.font.set('Missing Later Font')
        gate.set()
        deadline=time.monotonic()+10
        while not any(h.process is not None and h.process.poll() is None for h in app.manager.jobs.values()) and time.monotonic()<deadline:
            app.root.update();time.sleep(.01)
        assert any(h.process is not None and h.process.poll() is None for h in app.manager.jobs.values())
        app.infile.set('Z:\\switched_during_execution.h5');app.outfile.set(str(tmp_path/'out_D'));app.font.set('Another Missing Font')
        app.root.update();wait_jobs(app.manager,app)
        assert {h.state for h in app.manager.jobs.values()}=={'success'}
        for job_id,input_path,root,font in [(first,a,tmp_path/'out_A','Arial'),(second,b,tmp_path/'out_B','Times New Roman')]:
            h=app.manager.jobs[job_id]
            assert h.snapshot.input_path==str(input_path)
            record=next(root.rglob('analysis_manifest.json'))
            assert json.loads(record.read_text())['input_path']==str(input_path.resolve())
            render=next(root.rglob('render_manifest.json'))
            assert json.loads(render.read_text())['actual_font']==font
            assert list(root.rglob('IPF_X.png'))
            assert json.loads(h.snapshot.control_file.read_text())['state']=='success'
        assert not (tmp_path/'out_C').exists()
        assert not (tmp_path/'out_D').exists()
        assert all(storage.digest(p)==before[str(p)] for p in (a,b))
        assert not list(PROJECT.glob('IPF_*.png')) and not (PROJECT/'EBSD_paper_layout.png').exists()
    finally:
        gate.set()
        for job_id,h in app.manager.jobs.items():
            if not h.done.is_set():app.manager.cancel(job_id,force=True)
        app.root.destroy()

def test_cancel_before_popen_is_assigned_is_not_lost(tmp_path,monkeypatch):
    from gui import App
    app=App();app.root.withdraw();app.preferences=tmp_path/'prefs.json'
    source=raw(tmp_path/'A');gate=threading.Event();original=jobs.subprocess.Popen
    def delayed(*args,**kwargs):
        assert gate.wait(10)
        return original(*args,**kwargs)
    monkeypatch.setattr(jobs.subprocess,'Popen',delayed)
    try:
        app.infile.set(str(source));app.outfile.set(str(tmp_path/'out'));app.formats.set('png')
        app.start();selected=app.active_job.get();app.stop()
        assert app.manager.jobs[selected].process is None
        gate.set();wait_jobs(app.manager,app)
        handle=app.manager.jobs[selected]
        assert handle.state=='cancelled'
        assert json.loads(handle.snapshot.control_file.read_text())['state']=='cancelled'
        assert not list((tmp_path/'out').rglob('features.h5'))
    finally:gate.set();app.root.destroy()

def test_two_concurrent_render_processes_same_cache_keep_history(tmp_path):
    source=raw(tmp_path/'A');root=tmp_path/'out'
    assert main([str(source),'--output-root',str(root),'--no-render'])==0
    cache=next(root.rglob('features.h5'));before=storage.digest(cache)
    source.rename(source.with_suffix('.offline'))
    manager=JobManager(PROJECT)
    for font in ('Arial','Times New Roman'):
        manager.submit([sys.executable,'-u','-m','ebsd_autoplot','--cache',str(cache),'--font',font,
            '--formats','pdf','svg'],root,cache_path=str(cache),style=(font,))
    wait_jobs(manager)
    assert all(h.state=='success' for h in manager.jobs.values())
    records=list(root.rglob('render_manifest.json'));assert len(records)==2
    assert {json.loads(p.read_text())['actual_font'] for p in records}=={'Arial','Times New Roman'}
    assert len({p.parent.name for p in records})==2
    assert all((p.parent/'combined'/'EBSD_paper_layout.svg').is_file() for p in records)
    latest=json.loads(next(root.rglob('latest.json')).read_text())
    assert latest['render']['id'] in {p.parent.name for p in records}
    assert storage.digest(cache)==before

def test_process_launch_permission_failure_is_recorded(tmp_path,monkeypatch):
    def denied(*a,**kw):raise PermissionError('injected launch permission failure')
    monkeypatch.setattr(jobs.subprocess,'Popen',denied)
    manager=JobManager(PROJECT);snapshot=manager.submit([sys.executable,'-m','ebsd_autoplot'],tmp_path/'out')
    wait_jobs(manager,timeout=5)
    handle=manager.jobs[snapshot.job_id]
    assert handle.state=='failed'
    assert json.loads(snapshot.control_file.read_text())['state']=='failed'

def test_final_job_ledger_failure_still_releases_gui_waiters(tmp_path,monkeypatch):
    real=jobs.atomic_json;calls=[0]
    def deny_final(*args,**kwargs):
        calls[0]+=1
        if calls[0]>1:raise PermissionError('injected final job ledger failure')
        return real(*args,**kwargs)
    monkeypatch.setattr(jobs,'atomic_json',deny_final)
    monkeypatch.setattr(jobs.subprocess,'Popen',lambda *a,**kw:(_ for _ in ()).throw(PermissionError('launch denied')))
    manager=JobManager(PROJECT);snapshot=manager.submit([sys.executable,'-m','ebsd_autoplot'],tmp_path/'out')
    wait_jobs(manager,timeout=5)
    assert manager.jobs[snapshot.job_id].state=='failed'

def test_force_stop_marks_only_selected_job_cancelled_and_can_resume(tmp_path):
    source=raw(tmp_path/'A');root=tmp_path/'out';ready=tmp_path/'ready.json'
    script="""
import sys,time,json
from pathlib import Path
from ebsd_autoplot import pipeline
from ebsd_autoplot.cli import main
event=pipeline.event
def hold(kind,path,**kw):
    event(kind,path,**kw)
    if kw.get('stage')=='v1':
        Path(sys.argv[1]).write_text(json.dumps({'run':str(path)}))
        while True:time.sleep(.05)
pipeline.event=hold
raise SystemExit(main(sys.argv[2:]))
"""
    manager=JobManager(PROJECT)
    snap=manager.submit([sys.executable,'-u','-c',script,str(ready),str(source),'--output-root',str(root),
        '--no-render','--no-ipf'],root,input_path=str(source))
    handle=manager.jobs[snap.job_id]
    try:
        deadline=time.monotonic()+25
        while not ready.exists() and not handle.done.is_set() and time.monotonic()<deadline:time.sleep(.03)
        assert ready.exists()
        manager.cancel(snap.job_id,force=True);wait_jobs(manager,timeout=15)
        assert handle.state=='cancelled'
        mp=next(root.rglob('analysis_manifest.json'));assert json.loads(mp.read_text())['status']=='cancelled'
        cache=next(root.rglob('features.h5'));before=storage.digest(cache)
        pipeline.resume(mp);assert storage.digest(cache)==before
        assert json.loads(mp.read_text())['status']=='success'
    finally:
        if not handle.done.is_set():manager.cancel(snap.job_id,force=True)
