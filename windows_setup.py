"""Project-scoped Windows installation, repair and launch. UTF-8 logs."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import traceback

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')

ROOT = Path(__file__).resolve().parent
VENV = ROOT / '.venv'
PYTHON = VENV / 'Scripts' / 'python.exe'
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
LOG = None
STATE = {}

def safe(text):
    return re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@', r'\1<redacted>@', str(text))

def record(text):
    text = safe(text)
    print(text, flush=True)
    if LOG:
        LOG.write(text + '\n')
        LOG.flush()

def state(**values):
    STATE.update(values)
    STATE['updated'] = datetime.now().astimezone().isoformat()
    temp = ROOT / 'INSTALL_STATE.json.tmp'
    temp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(ROOT / 'INSTALL_STATE.json')

def run(argv, required=True, timeout=None):
    argv = [str(x) for x in argv]
    record('COMMAND: ' + subprocess.list2cmdline(argv))
    state(current_command=safe(subprocess.list2cmdline(argv)), step='running')
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PIP_DISABLE_PIP_VERSION_CHECK='1')
    if timeout is not None:
        completed = subprocess.run(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', timeout=timeout)
        record(completed.stdout)
    else:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        lines = []
        for line in process.stdout:
            lines.append(line)
            record(line.rstrip('\n'))
        completed = subprocess.CompletedProcess(argv, process.wait(), ''.join(lines))
    record(f'EXIT: {completed.returncode}')
    if completed.returncode == 0:
        STATE.setdefault('successful_commands', []).append(safe(subprocess.list2cmdline(argv)))
    state(last_exit_code=completed.returncode)
    if required and completed.returncode:
        raise RuntimeError(f'Command failed ({completed.returncode}); see {STATE["log"]}')
    return completed

def compatible(path):
    if not Path(path).is_file():
        return False
    try:
        result = run([path, '-c', 'import sys,struct; print(sys.executable); print(sys.version); '
                      'assert sys.version_info[:2]==(3,12) and struct.calcsize("P")==8'],
                     required=False, timeout=30)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        record(str(exc))
        return False

def ensure_environment():
    if compatible(PYTHON):
        return
    base = Path(sys.executable)
    if VENV in base.parents or not compatible(base):
        raise RuntimeError('Run INSTALL/REPAIR with a base Python 3.12 x64, outside .venv.')
    if VENV.exists():
        # Fail closed if Windows cannot establish whether this environment is in use.
        query = "$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process | Select-Object ExecutablePath,CommandLine | ConvertTo-Json -Compress"
        result = run(['powershell.exe', '-NoProfile', '-Command', query], timeout=30)
        processes = json.loads(result.stdout or '[]')
        if isinstance(processes, dict):
            processes = [processes]
        for proc in processes:
            if str(VENV).lower() in ((proc.get('ExecutablePath') or '') + (proc.get('CommandLine') or '')).lower():
                raise RuntimeError('The old .venv is in use. Close its processes before repair.')
        target = ROOT / ('.venv.backup_' + STAMP)
        assert target.parent.resolve() == ROOT and VENV.resolve().parent == ROOT
        VENV.rename(target)
        record(f'Preserved previous environment: {target}')
    run([base, '-m', 'venv', VENV])
    if not compatible(PYTHON):
        raise RuntimeError('New environment verification failed')

def install(action):
    ensure_environment()
    pip = run([PYTHON, '-m', 'pip', '--version'], required=False)
    if pip.returncode:
        run([PYTHON, '-m', 'ensurepip', '--upgrade'])
    # A healthy repair does not reinstall or upgrade anything.
    if action == 'repair':
        health = run([PYTHON, ROOT / 'verify_install.py'], required=False)
        check = run([PYTHON, '-m', 'pip', 'check'], required=False)
        if health.returncode == check.returncode == 0:
            state(step='verified', next='START_WINDOWS.cmd', status='environment_healthy')
            record('Environment healthy; no packages changed.')
            return
    options = ['--index-url', 'https://pypi.org/simple', '--timeout', '60', '--retries', '2']
    if action == 'install':
        run([PYTHON, '-m', 'pip', 'install', *options, '--upgrade', 'pip'])
    run([PYTHON, '-m', 'pip', 'install', *options, '--only-binary=:all:', 'PyCifRW==5.0.1'])
    manifest = ROOT / 'requirements-windows-py312.lock.txt'
    if not manifest.exists():
        manifest = ROOT / 'requirements.txt'
    repair_flags = ['--force-reinstall'] if action == 'repair' else []
    run([PYTHON, '-m', 'pip', 'install', *options, '--prefer-binary', *repair_flags, '-r', manifest])
    run([PYTHON, '-m', 'pip', 'install', *options, 'pytest'])
    run([PYTHON, '-m', 'pip', 'check'])
    run([PYTHON, ROOT / 'verify_install.py'])
    run([PYTHON, '-m', 'pytest', 'tests', '-q'])
    frozen = run([PYTHON, '-m', 'pip', 'freeze'])
    lines = frozen.stdout.splitlines()
    if any(' @ ' in line or line.startswith('-e') for line in lines):
        raise RuntimeError('Non-portable dependency lock; inspect before saving')
    if not (ROOT / 'requirements-windows-py312.lock.txt').exists():
        (ROOT / 'requirements-windows-py312.lock.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    state(step='verified', next='Run real ROI smoke test, then START_WINDOWS.cmd', status='environment_tests_passed')
    record('Environment and automated checks passed. See INSTALL_REPORT.md for full acceptance status.')

def main():
    global LOG, STATE
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['install','repair','start'])
    parser.add_argument('--unattended', action='store_true')
    parser.add_argument('--smoke-exit-ms', type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    for directory in ['logs','backups']:
        (ROOT / directory).mkdir(exist_ok=True)
    previous = ROOT / 'INSTALL_STATE.json'
    if previous.exists():
        STATE = json.loads(previous.read_text(encoding='utf-8-sig'))
    log_path = ROOT / 'logs' / f'{args.action}_{STAMP}.log'
    with log_path.open('w', encoding='utf-8') as log:
        LOG = log
        state(action=args.action, step='starting', log=str(log_path), project=str(ROOT),
              python=str(PYTHON), windows=platform.platform(), next='Read current log if interrupted')
        try:
            if args.action == 'start':
                if not compatible(PYTHON):
                    raise RuntimeError('Python 3.12 x64 .venv missing/broken; run REPAIR_WINDOWS.cmd')
                ready = ROOT / 'logs' / f'gui_ready_{STAMP}.json'
                state(step='gui_running', ready_file=str(ready))
                argv = [PYTHON, '-u', ROOT / 'gui.py', '--ready-file', ready]
                if args.smoke_exit_ms:
                    argv.extend(['--smoke-exit-ms', args.smoke_exit_ms])
                run(argv)
                if not ready.exists():
                    raise RuntimeError('GUI exited without a Tk event-loop ready signal')
                state(step='gui_closed', next='START_WINDOWS.cmd')
            else:
                install(args.action)
            return 0
        except Exception:
            record(traceback.format_exc())
            state(step='failed', next='Inspect log and rerun REPAIR_WINDOWS.cmd')
            return 1

if __name__ == '__main__':
    raise SystemExit(main())
