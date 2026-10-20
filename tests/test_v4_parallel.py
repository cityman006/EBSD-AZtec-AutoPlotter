"""Real child-process barriers and admission/cancellation invariants."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch
import pytest
from ebsd_autoplot.batch import create_batch, execute_batch, request_cancel
from ebsd_autoplot.storage import atomic_json
from test_pipeline import write_synth


CHILD = r'''
import json,os,sys,time
from pathlib import Path
idx,control,root,mode,attempt=sys.argv[1:]
root=Path(root);control=Path(control);tag=idx+'_'+attempt
info={'index':idx,'attempt':attempt,'start':time.monotonic(),
      'threads':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')}}
(root/(tag+'.start.json')).write_text(json.dumps(info))
print('ready '+idx,flush=True)
code=0
try:
 if mode=='barrier':
  peer=str(int(idx)^1)+'_'+attempt+'.start.json';deadline=time.monotonic()+4
  while not (root/peer).exists():
   if time.monotonic()>deadline:raise RuntimeError('no concurrent peer reached process barrier')
   time.sleep(.01)
 if mode=='cancel':
  while not json.loads(control.read_text())['cancel']:time.sleep(.01)
  code=130
 elif mode=='flood':
  for i in range(1500):print('payload '+idx+' '+('x'*256),flush=True)
 elif mode=='failure':code=7
 else:time.sleep(.15)
finally:
 info['end']=time.monotonic();info['exit']=code
 (root/(tag+'.end.json')).write_text(json.dumps(info))
sys.exit(code)
'''


def fixture_batch(tmp_path, count, *, concurrency=2, reservation_mb=30, budget_mb=70):
    run=tmp_path/'batch';run.mkdir()
    records=[{'input':str(i),'source_name':f'sample{i}','scan_path':'/1/EBSD',
              'scan_id':f'scan{i}','source_stamp':[], 'status':'queued',
              'estimated_total_bytes':reservation_mb*1024**2,'estimated_block_bytes':1}
             for i in range(count)]
    atomic_json(run/'batch_manifest.json',{'schema_version':2,'status':'queued','records':records,
        'options':[], 'concurrency':concurrency,'memory_budget_mb':budget_mb,'output_root':str(tmp_path)})
    script=tmp_path/'controlled_child.py';script.write_text(CHILD,encoding='utf-8')
    evidence=tmp_path/'process_evidence';evidence.mkdir()
    return run,script,evidence


def real_children(script,evidence,mode='complete'):
    original=subprocess.Popen;children=[];attempts={}
    def factory(argv,**kwargs):
        index=argv[4];attempts[index]=attempts.get(index,0)+1
        control=argv[argv.index('--control-file')+1]
        child_mode=mode(index,attempts[index]) if callable(mode) else mode
        proc=original([sys.executable,'-u',str(script),index,control,str(evidence),child_mode,str(attempts[index])],**kwargs)
        children.append(proc);return proc
    return factory,children


def maximum_overlap(evidence):
    intervals=[json.loads(path.read_text()) for path in evidence.glob('*.end.json')]
    events=[]
    for record in intervals:events.extend([(record['start'],1),(record['end'],-1)])
    active=peak=0
    for _,delta in sorted(events):active+=delta;peak=max(peak,active)
    return peak,intervals


def cleanup(children):
    for child in children:
        if child.poll() is None:child.terminate();child.wait()


def test_real_process_barrier_proves_parallel_two_and_concurrency_cap(tmp_path):
    run,script,evidence=fixture_batch(tmp_path,4,concurrency=2)
    factory,children=real_children(script,evidence,'barrier')
    writer=atomic_json;snapshots=[];owner=threading.get_ident();messages=[]
    def checked_writer(path,payload):
        assert threading.get_ident()==owner  # Pipe readers never mutate persistent state.
        if Path(path).name=='batch_manifest.json':snapshots.append(payload.get('resource_state',{}).copy())
        writer(path,payload)
    try:
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory), patch('ebsd_autoplot.batch.atomic_json',side_effect=checked_writer):
            result=execute_batch(run,progress=messages.append)
        assert result['status']=='success' and all(child.poll()==0 for child in children)
        peak,intervals=maximum_overlap(evidence)
        assert peak==2 and len(intervals)==4
        assert any(row['active_children']==2 for row in snapshots)
        assert all(row['active_children']<=2 and row['reserved_bytes']<=70*1024**2 for row in snapshots)
        assert all(set(row['threads'].values())=={'1'} for row in intervals)
        assert all(any(line.startswith(f'[sample{i} | scan{i}] ready ') for line in messages) for i in range(4))
        assert (run/'scan0.log').read_text().startswith('ready 0')  # Persist original per-child stream.
    finally:cleanup(children)


def test_total_active_memory_admission_serializes_even_when_slots_available(tmp_path):
    run,script,evidence=fixture_batch(tmp_path,3,concurrency=3,reservation_mb=40,budget_mb=70)
    factory,children=real_children(script,evidence)
    try:
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory):result=execute_batch(run,progress=lambda _:None)
        assert result['status']=='success'
        assert maximum_overlap(evidence)[0]==1
        assert result['resource_state']['reserved_bytes']==0
    finally:cleanup(children)


def test_cancel_reaches_every_active_child_and_queued_remains_resumable(tmp_path):
    run,script,evidence=fixture_batch(tmp_path,3,concurrency=2)
    factory,children=real_children(script,evidence,lambda i,attempt:'cancel' if attempt==1 else 'complete')
    ready=[];resumed_children=[]
    def progress(line):
        if line.split('] ',1)[-1].startswith('ready '):ready.append(line)
        if len(ready)==2:request_cancel(run)
    try:
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory):first=execute_batch(run,progress=progress)
        assert first['status']=='cancelled'
        assert [row['status'] for row in first['records']]==['cancelled','cancelled','queued']
        assert len(children)==2 and all(child.poll()==130 for child in children)
        assert all(json.loads((run/f'scan{i}.control.json').read_text())['cancel'] for i in (0,1))
        original={p.name:p.read_bytes() for p in evidence.glob('*.end.json')}
        (run/'cancel.json').unlink()  # Explicit Resume, same behaviour as GUI.
        # Previously queued job has no first attempt, so make every retry complete.
        # Keep evidence names unique across resumes without modifying old records.
        resumed_evidence=evidence/'resume';resumed_evidence.mkdir()
        resumed_factory,resumed_children=real_children(script,resumed_evidence,'complete')
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=resumed_factory):second=execute_batch(run,progress=lambda _:None)
        assert second['status']=='success' and all(row['status']=='success' for row in second['records'])
        assert all((evidence/name).read_bytes()==data for name,data in original.items())
    finally:cleanup(children);cleanup(resumed_children)


def test_parallel_failures_continue_and_independent_streams_do_not_deadlock(tmp_path):
    run,script,evidence=fixture_batch(tmp_path,4,concurrency=2)
    factory,children=real_children(script,evidence,lambda i,_:'failure' if i=='0' else 'flood')
    try:
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory):result=execute_batch(run,progress=lambda _:None)
        assert [row['status'] for row in result['records']]==['failed','success','success','success']
        assert result['status']=='failed' and all(child.poll() is not None for child in children)
        assert all((run/f'scan{i}.log').stat().st_size>300000 for i in (1,2,3))
    finally:cleanup(children)


def test_parallel_consumer_exception_cleans_every_affected_owned_child(tmp_path):
    run,script,evidence=fixture_batch(tmp_path,2,concurrency=2)
    factory,children=real_children(script,evidence,'cancel')
    def broken(_):raise RuntimeError('consumer rejected event')
    try:
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory):result=execute_batch(run,progress=broken)
        assert result['status']=='failed' and all(row['status']=='failed' for row in result['records'])
        assert all(child.poll() is not None for child in children)
        assert all(json.loads((run/f'scan{i}.control.json').read_text())['cancel'] for i in (0,1))
    finally:cleanup(children)


def test_parallel_keyboard_interrupt_cleans_all_active_not_queued(tmp_path):
    run,script,evidence=fixture_batch(tmp_path,3,concurrency=2)
    factory,children=real_children(script,evidence,'cancel');ready=[]
    def interrupt(line):
        if line.split('] ',1)[-1].startswith('ready '):ready.append(line)
        if len(ready)==2:raise KeyboardInterrupt('controlled supervisor interruption')
    try:
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory),pytest.raises(KeyboardInterrupt):execute_batch(run,progress=interrupt)
        result=json.loads((run/'batch_manifest.json').read_text())
        assert result['status']=='interrupted'
        assert [row['status'] for row in result['records']]==['interrupted','interrupted','queued']
        assert len(children)==2 and all(child.poll() is not None for child in children)
    finally:cleanup(children)


@pytest.mark.parametrize('mode',['v1v2','v4_quantitative'])
def test_resource_snapshot_includes_baseline_mode_roi_and_legacy_default(tmp_path,mode):
    write_synth(tmp_path/'raw',h=8,w=8)
    source=tmp_path/'raw/sample.h5oina'
    path=create_batch([source],tmp_path/'out',['--analysis',mode,'--roi','0','0','4','5','--no-render'],concurrency=3)
    data=json.loads((path/'batch_manifest.json').read_text());estimate=data['records'][0]['resource_estimate']
    assert data['concurrency']==3 and data['threads_per_child']==1
    assert estimate['baseline_bytes']>0 and estimate['block_bytes']>0
    assert estimate['grain_roi_pixels']==20 and estimate['grain_roi_bytes']>0 and not estimate['hard_os_limit']
    assert estimate['total_bytes']>=estimate['baseline_bytes']+estimate['block_bytes']+estimate['grain_roi_bytes']
    for invalid in (0,9,1.5,True):
        with pytest.raises(ValueError,match='Concurrency'):create_batch([source],tmp_path/'out',concurrency=invalid)
    with pytest.raises(ValueError,match='identity'):
        create_batch([source],tmp_path/'out',['--output-r','unexpected'])
    legacy_parent=tmp_path/'legacy';legacy_parent.mkdir()
    legacy,script,evidence=fixture_batch(legacy_parent,1)
    old=json.loads((legacy/'batch_manifest.json').read_text());old.pop('concurrency');old['records'][0].pop('estimated_total_bytes')
    atomic_json(legacy/'batch_manifest.json',old)
    factory,children=real_children(script,evidence)
    try:
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=factory):result=execute_batch(legacy,progress=lambda _:None)
        assert result['status']=='success' and maximum_overlap(evidence)[0]==1
    finally:cleanup(children)


def test_two_real_scientific_children_cache_recovery_and_budget_rejection(tmp_path):
    paths=[]
    for i in range(2):write_synth(tmp_path/f'raw{i}',h=4,w=4);paths.append(tmp_path/f'raw{i}/sample.h5oina')
    run=create_batch(paths,tmp_path/'output',['--analysis','v1','--no-render','--no-ipf'],concurrency=2,memory_budget_mb=1024)
    first=execute_batch(run,progress=lambda _:None)
    assert first['status']=='success'
    original=[row['run_path'] for row in first['records']]
    assert len(set(original))==2
    with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=AssertionError('verified cache must not rerun')):
        second=execute_batch(run,progress=lambda _:None)
    assert second['status']=='success' and [row['run_path'] for row in second['records']]==original
    denied=create_batch([paths[0]],tmp_path/'denied',['--no-render','--no-ipf'],memory_budget_mb=16,concurrency=2)
    with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=AssertionError('over-budget child must not launch')):
        rejected=execute_batch(denied,progress=lambda _:None)
    assert rejected['status']=='failed' and 'total child memory' in rejected['records'][0]['error']


def test_cancelled_batch_module_exit_is_130(tmp_path):
    run,_,_=fixture_batch(tmp_path,1);request_cancel(run)
    result=subprocess.run([sys.executable,'-m','ebsd_autoplot.batch',str(run)],capture_output=True,text=True)
    assert result.returncode==130


def test_resume_refuses_still_running_previous_child_without_stopping_it(tmp_path):
    from ebsd_autoplot.pipeline import process_token
    run,_,_=fixture_batch(tmp_path,1)
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        data=json.loads((run/'batch_manifest.json').read_text())
        data['records'][0].update(status='running',child_pid=child.pid,child_token=process_token(child.pid))
        atomic_json(run/'batch_manifest.json',data)
        with patch('ebsd_autoplot.batch.subprocess.Popen',side_effect=AssertionError('must not duplicate an active analysis')):
            with pytest.raises(ValueError,match='still running'):execute_batch(run,progress=lambda _:None)
        assert child.poll() is None
        assert json.loads((run/'batch_manifest.json').read_text())['records'][0]['status']=='running'
    finally:child.terminate();child.wait()


def test_cli_interruption_exit_one_and_resource_budget_must_be_finite(tmp_path):
    from ebsd_autoplot.batch import main
    with patch('ebsd_autoplot.batch.execute_batch',side_effect=KeyboardInterrupt):
        assert main(['interrupted-batch'])==1
    for value in (float('nan'),float('inf')):
        with pytest.raises(ValueError,match='Memory budget'):create_batch([],tmp_path,memory_budget_mb=value)
