"""Resumeable bounded acceptance run for the repaired Windows installation."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import windows_setup as setup

root=Path(__file__).resolve().parent
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
setup.STATE=json.loads((root/'INSTALL_STATE.json').read_text(encoding='utf-8-sig'))
with (root/'logs'/f'acceptance_{stamp}.log').open('w',encoding='utf-8') as log:
    setup.LOG=log
    setup.state(step='acceptance',log=str(root/'logs'/f'acceptance_{stamp}.log'))
    setup.run([setup.PYTHON,'-m','ebsd_autoplot','--help'])
    from ebsd_autoplot.io import H5Source
    source=root/'demo'/'real_montage_roi.h5oina'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    assert before=='192ea35e7923130f74d906a3ccf7fa98dd38e95b6984161dad7b5a04927abe52'
    info=H5Source(source)
    assert info.nx>=64 and info.ny>=64
    setup.record(json.dumps(info.header,ensure_ascii=False,indent=2))
    runroot=root/'output'/('installation_smoke_'+stamp)
    runroot.mkdir(parents=True,exist_ok=False)
    copied=runroot/'真实样本 副本.h5oina'
    shutil.copy2(source,copied)
    output=runroot/'中文 路径'
    setup.state(step='real_roi',smoke_output=str(output),raw_sha256=before)
    setup.run([setup.PYTHON,'-m','ebsd_autoplot',copied,'--analysis','v1v2','--out',output,
               '--batch','8','--roi','0','0','64','64','--texture','--mtex-roi','0','0','64','64'])
    setup.run([setup.PYTHON,root/'acceptance_check.py',copied,output])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    setup.state(step='gui_smoke',core_environment='passed',next='GUI and repair checks')
    setup.run(['cmd.exe','/d','/c',str(root/'START_WINDOWS.cmd'),'--unattended','--smoke-exit-ms','1500'])
    ready_path=max((root/'logs').glob('gui_ready_*.json'),key=lambda p:p.stat().st_mtime)
    ready=json.loads(ready_path.read_text(encoding='utf-8'))
    assert all(ready[key] for key in ['mapped','visible','event_loop','file_selection_command']),ready
    setup.record('GUI validated: '+json.dumps(ready))
    lock=root/'requirements-windows-py312.lock.txt'
    locked=lock.read_bytes()
    setup.run(['cmd.exe','/d','/c',str(root/'REPAIR_WINDOWS.cmd'),'--unattended'])
    assert lock.read_bytes()==locked
    setup.state(step='acceptance_passed',core_environment='passed',gui_smoke=ready,
                smoke_output=str(output),next='START_WINDOWS.cmd and final report')
    print('Acceptance output:',output)
