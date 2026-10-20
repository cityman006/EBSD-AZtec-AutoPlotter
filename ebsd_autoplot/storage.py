"""Atomic artifacts, cancellation and OS-owned locks (no stale lock deletion)."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import hashlib
import inspect
import json
import os
import time
import uuid

_cancel = ContextVar('cancel', default=None)

class Cancelled(RuntimeError):
    pass

def long_path(path):
    path=Path(path).expanduser().absolute()
    value=str(path)
    if os.name=='nt' and not value.startswith('\\\\?\\'):
        value='\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value
    return Path(value)

def canonical_path(path):
    value=str(Path(path).expanduser().resolve())
    if value.startswith('\\\\?\\UNC\\'):value='\\\\'+value[8:]
    elif value.startswith('\\\\?\\'):value=value[4:]
    return Path(value)

def check_cancelled():
    checker = _cancel.get()
    if checker is not None and checker():
        raise Cancelled('Cancellation requested; committed checkpoints are retained')

@contextmanager
def cancellation(checker):
    token = _cancel.set(checker)
    try: yield
    finally: _cancel.reset(token)

def digest(path):
    h = hashlib.sha256()
    with long_path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def atomic_json(path, payload):
    path = long_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.j' + uuid.uuid4().hex[:12] + '.tmp')
    created = False
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as f:
            created = True
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        # Windows readers may briefly open control JSON without delete sharing.
        # Retain the complete old record until one atomic replacement succeeds;
        # never fall back to truncating or writing the destination in place.
        deadline = time.monotonic() + 2.
        delay = .01
        while True:
            try:
                os.replace(temp, path)
                break
            except PermissionError as exc:
                # winerror evidence keeps ordinary POSIX/other permission errors
                # immediate; retry only documented Windows access/sharing locks.
                if getattr(exc, 'winerror', None) not in (5, 32, 33):raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:raise
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, .1)
    finally:
        if created:temp.unlink(missing_ok=True)  # Only this call's private temp.

@contextmanager
def file_lock(path, timeout=30):
    path = long_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    # Byte-range locks may extend beyond EOF. Initializing the byte before
    # taking the lock would itself race a process which already owns it.
    deadline = time.monotonic() + timeout
    while True:
        try:
            f.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() >= deadline:
                f.close(); raise TimeoutError(f'Output is busy: {path}')
            time.sleep(.05)
    try: yield
    finally:
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()

def inside(root, relative):
    root = long_path(root).resolve(); path = (root / relative).resolve()
    if root not in path.parents: raise ValueError(f'Path escapes task directory: {relative}')
    return path

def _publish(temp, target):
    """Create a final filename atomically; never replace an existing artifact."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with temp.open('r+b') as f: os.fsync(f.fileno())
    if os.name == 'nt':
        os.rename(temp,target)  # Windows rename is atomic and refuses an existing destination.
    else:
        os.link(temp,target)
        temp.unlink()

def finish_plan(root, plan_path):
    root = long_path(root).resolve()
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    # Verify the complete plan before making *any* additional final file visible.
    for item in plan['files']:
        final = inside(root, item['target']); temp = inside(root, item['temp'])
        chosen = final if final.exists() else temp
        if not chosen.is_file() or digest(chosen) != item['sha256']:
            raise ValueError(f'Unverifiable transaction artifact: {chosen}')
    for item in plan['files']:
        check_cancelled()
        final = inside(root, item['target']); temp = inside(root, item['temp'])
        if not final.exists(): _publish(temp, final)
    plan['status'] = 'committed'
    atomic_json(plan_path, plan)
    return plan

def recover_transactions(root):
    root = long_path(root)
    for plan in sorted(root.glob('.transactions/*.json')):
        data = json.loads(plan.read_text(encoding='utf-8'))
        if data['status'] == 'prepared': finish_plan(root, plan)

def validate_receipt(root, operation):
    root = long_path(root)
    matches = []
    for plan in root.glob('.transactions/*.json'):
        data = json.loads(plan.read_text(encoding='utf-8'))
        if data.get('operation') == operation and data.get('status') == 'committed': matches.append(data)
    if not matches: return False
    for item in matches[-1]['files']:
        final = inside(root, item['target'])
        if not final.is_file() or digest(final) != item['sha256']:
            raise ValueError(f'Checkpoint changed or incomplete: {final}')
    return True

def transactional_output(parameter='outdir', *, file_target=False):
    """Redirect an output producer into its own staging tree, then commit its files.

    Prepared journals allow explicit Resume after a kill in the commit phase.
    Computation killed before preparation is restarted in a new staging tree.
    """
    def decorate(func):
        sig = inspect.signature(func)
        @wraps(func)
        def call(*args, **kwargs):
            bound = sig.bind(*args, **kwargs); bound.apply_defaults()
            target = long_path(bound.arguments[parameter]).resolve()
            # A directory named 'path' (IPF keys) is distinct from an output file.
            root = target.parent if file_target else target
            root.mkdir(parents=True, exist_ok=True)
            operation = func.__module__ + '.' + func.__name__
            with file_lock(root / '.artifact.lock'):
                check_cancelled()
                txn = uuid.uuid4().hex[:12]
                stage = root / '.staging' / txn; stage.mkdir(parents=True)
                bound.arguments[parameter] = stage / target.name if file_target else stage
                result = func(*bound.args, **bound.kwargs)
                check_cancelled()
                files = []
                for temp in sorted(stage.rglob('*')):
                    if not temp.is_file(): continue
                    rel = temp.relative_to(stage)
                    final = inside(root, rel)
                    if final.exists(): raise FileExistsError(f'Refusing to replace artifact {final}')
                    files.append({'temp':temp.relative_to(root).as_posix(), 'target':rel.as_posix(),
                                  'sha256':digest(temp), 'size':temp.stat().st_size})
                if not files: raise ValueError(f'Output operation produced no files: {operation}')
                plan = root / '.transactions' / (txn + '.json')
                atomic_json(plan, {'schema_version':1,'operation':operation,'status':'prepared','files':files})
                finish_plan(root, plan)
                if isinstance(result, Path): return root / result.relative_to(stage)
                return result
        return call
    return decorate
