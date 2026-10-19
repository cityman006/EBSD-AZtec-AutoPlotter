"""GUI process jobs: immutable launch snapshots and per-job cancellation controls."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import queue
import subprocess
import threading
import uuid
from .storage import atomic_json, long_path
from .pipeline import recover_interrupted

@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    argv: tuple[str,...]
    output_root: Path
    input_path: str
    cache_path: str
    style: tuple
    control_file: Path

@dataclass
class JobHandle:
    snapshot: JobSnapshot
    process: object = None
    state: str = 'queued'
    returncode: int | None = None
    targets: set = field(default_factory=set)
    cancel_requested: bool = False
    forced: bool = False
    done: threading.Event = field(default_factory=threading.Event)

class JobManager:
    def __init__(self,cwd,events=None):
        self.cwd=str(Path(cwd).resolve());self.events=events if events is not None else queue.Queue()
        self.jobs={};self.lock=threading.Lock()
    def submit(self,argv,output_root,*,input_path='',cache_path='',style=()):
        root=Path(output_root).expanduser().resolve()
        job_id='job_'+uuid.uuid4().hex[:12]
        control=root/'.job_control'/(job_id+'.json')
        atomic_json(control,{'cancel':False,'state':'queued','argv':list(argv),'input_path':str(input_path),
            'cache_path':str(cache_path),'output_root':str(root),'style':list(style)})
        snapshot=JobSnapshot(job_id,tuple(str(x) for x in argv)+('--control-file',str(control)),root,
            str(input_path),str(cache_path),tuple(style),control)
        handle=JobHandle(snapshot)
        with self.lock:self.jobs[job_id]=handle  # Reserve before the worker can start.
        threading.Thread(target=self._run,args=(handle,),daemon=True).start()
        return snapshot
    def _run(self,handle):
        snap=handle.snapshot
        try:
            proc=subprocess.Popen(snap.argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',cwd=self.cwd)
            with self.lock:handle.process=proc;handle.state='running'
            if handle.forced:proc.terminate()
            for line in proc.stdout:
                if line.startswith('JOB_EVENT '):
                    try:
                        event=json.loads(line[len('JOB_EVENT '):]);handle.targets.add(str(Path(event['path']).resolve()))
                    except (ValueError,KeyError):pass
                self.events.put((snap.job_id,'line',line))
            code=proc.wait();handle.returncode=code
            handle.state='success' if code==0 else ('cancelled' if code==130 or handle.cancel_requested else 'failed')
            if code!=0:
                recovered=recover_interrupted(snap.output_root)
                if handle.forced:
                    for mp in recovered:
                        mp=Path(mp)
                        if str(long_path(mp.parent).resolve()) in handle.targets:
                            data=json.loads(mp.read_text(encoding='utf-8'))
                            data.update(status='cancelled',error='Explicit force stop; Resume can verify retained checkpoints')
                            atomic_json(mp,data)
        except BaseException as exc:
            handle.state='failed';self.events.put((snap.job_id,'line',f'FAILED: {type(exc).__name__}: {exc}\n'))
        finally:
            try:
                with self.lock:
                    data=json.loads(snap.control_file.read_text(encoding='utf-8'))
                    data.update(state=handle.state,returncode=handle.returncode,targets=sorted(handle.targets))
                    atomic_json(snap.control_file,data)
            except (OSError,ValueError) as exc:
                handle.state='failed';self.events.put((snap.job_id,'line',f'FAILED to persist job status: {exc}\n'))
            finally:
                handle.done.set();self.events.put((snap.job_id,'done',handle.state))
    def cancel(self,job_id,*,force=False):
        with self.lock:
            handle=self.jobs[job_id]
            if handle.done.is_set():return
            handle.cancel_requested=True
            data=json.loads(handle.snapshot.control_file.read_text(encoding='utf-8'));data['cancel']=True
            atomic_json(handle.snapshot.control_file,data)
            if force:
                handle.forced=True
                if handle.process is not None and handle.process.poll() is None:handle.process.terminate()
