"""Persisted, sequential batches. Each child uses the existing isolated CLI pipeline."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import threading
from datetime import datetime
from .discovery import discover_scans, ROWS
from .io import H5Source
from .storage import atomic_json, file_lock, validate_receipt


def inventory(paths):
    """Resolve montage directories once, while retaining independent files/scans."""
    records, seen, expanded = [], set(), []
    for value in paths:
        p = Path(value).expanduser().resolve()
        if p.is_file() and (ROWS.match(p.name) or p.suffix.lower() == '.json'):
            expanded.append(p.parent)
        elif p.is_dir():
            # Only folders with shards/scan manifests are one discovery unit.
            manifests = []
            for jp in p.glob('*.json'):
                try:
                    if isinstance(json.loads(jp.read_text(encoding='utf-8-sig')).get('scans'), list):
                        manifests.append(jp)
                except (ValueError, AttributeError):
                    pass
            if manifests or any(ROWS.match(x.name) for x in p.glob('*.h5')):
                expanded.append(p)
            else:
                expanded.extend(sorted(x for x in p.iterdir() if x.suffix.lower() in ('.h5', '.h5oina')))
        else:
            expanded.append(p)
    for path in dict.fromkeys(expanded):
        for spec in discover_scans(path):
            key = (tuple((str(p.resolve()), a, b) for p, a, b in spec.parts), spec.scan_path, spec.scan_tag)
            if key in seen:
                continue
            seen.add(key)
            src = H5Source(spec)
            stats = [{'path': str(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                     for p, _, _ in spec.parts]
            ident = hashlib.sha256(json.dumps([stats, spec.scan_path, spec.scan_tag], sort_keys=True).encode()).hexdigest()
            records.append({'input': str(spec.input_path), 'source_name': spec.source_name,
                'scan_path': spec.scan_path, 'scan_id': ident[:16], 'source_stamp': stats,
                'scan_tag': spec.scan_tag,
                'nx': src.nx, 'ny': src.ny, 'dx_um': src.sx, 'dy_um': src.sy,
                'phases': src.phase_specs, 'parts': len(spec.parts), 'status': 'queued',
                'estimated_cache_bytes': src.nx * src.ny * 48,
                'estimated_block_bytes': src.nx * 32 * 160})
    if not records:
        raise ValueError('No supported scans in the selected inputs')
    return records


def create_batch(paths, output_root, argv=(), *, memory_budget_mb=2048):
    if memory_budget_mb < 16:
        raise ValueError('Memory budget must be at least 16 MB')
    # Disallow arguments that could redirect a frozen batch snapshot.
    reserved = {'--output-root', '--out', '--scan', '--source-name', '--scan-tag', '--control-file', '--cache', '--resume'}
    if any(str(x).split('=')[0] in reserved for x in argv):
        raise ValueError('Batch options cannot override input/output/scan identity')
    records = inventory(paths)
    rows = int(argv[list(argv).index('--batch')+1]) if '--batch' in argv else 32
    if rows<1: raise ValueError('Block rows must be positive')
    for r in records:r['estimated_block_bytes']=r['nx']*rows*160
    root = Path(output_root).expanduser().resolve()
    run = root / 'batches' / ('batch_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    # A repeat batch may reference only a matching, independently verified run.
    for previous in sorted((root/'batches').glob('*/batch_manifest.json'),reverse=True):
        try:
            old=json.loads(previous.read_text(encoding='utf-8'))
            if old['options']!=list(argv):continue
            indexed={r['scan_id']:r for r in old['records'] if r['status']=='success'}
            for r in records:
                cached=indexed.get(r['scan_id'])
                if cached and _verified_record(cached):
                    r.update(status='success',returncode=0,run_path=cached['run_path'],cache_origin=old['batch_id'])
        except (ValueError,KeyError,OSError):continue
    data = {'schema_version': 1, 'batch_id': run.name, 'status': 'queued',
            'output_root': str(root), 'options': list(argv), 'concurrency': 1,
            'memory_budget_mb': memory_budget_mb, 'records': records, 'cancel': False}
    atomic_json(run / 'batch_manifest.json', data)
    return run


def request_cancel(run):
    atomic_json(Path(run) / 'cancel.json', {'cancel': True})


def _reports(run, data):
    fields = ['source_name', 'scan_id', 'nx', 'ny', 'dx_um', 'dy_um', 'status', 'returncode', 'run_path', 'error']
    for name, rows in [('batch_summary.csv', data['records']),
                       ('batch_failures.csv', [r for r in data['records'] if r['status'] == 'failed'])]:
        temp = run / ('.' + name + '.tmp')
        with temp.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        os.replace(temp, run / name)


def _verified_record(record):
    try:
        mp = Path(record['run_path']) / 'analysis_manifest.json'
        m = json.loads(mp.read_text(encoding='utf-8'))
        if m['status'] != 'success': return False
        for item in record['source_stamp']:
            st = Path(item['path']).stat()
            if (st.st_size, st.st_mtime_ns) != (item['size'], item['mtime_ns']): return False
        if not validate_receipt(mp.parent / 'analysis_cache', 'ebsd_autoplot.compute.prepare'): return False
        from .pipeline import _receipt
        cache = mp.parent / 'analysis_cache'
        operations = {'v2': (cache/'v2', 'ebsd_autoplot.v2.run_v2'),
            'v3_texture': (cache/'v3/texture/results', 'ebsd_autoplot.texture_v3.analyze_texture'),
            'v3_schmid': (cache/'v3/schmid/results', 'ebsd_autoplot.schmid.analyze_schmid'),
            'v3_variants': (cache/'v3/variants/results', 'ebsd_autoplot.alpha_variants.analyze_alpha_variants'),
            'v3_prior_beta': (cache/'v3/prior_beta/results', 'ebsd_autoplot.prior_beta.reconstruct_prior_beta')}
        for stage in m.get('checkpoints', {}):
            if stage in operations and not _receipt(*operations[stage]): return False
        if not m['options'].get('no_render', False):
            from .pipeline import _dataset_for
            if not validate_receipt(_dataset_for(mp.parent)/m['render_path'], 'ebsd_autoplot.render.maps'): return False
        return True
    except (KeyError, ValueError, OSError):
        return False


def execute_batch(run, *, progress=print):
    run = Path(run).resolve(); mp = run / 'batch_manifest.json'
    with file_lock(run / '.batch.lock', timeout=.1):
        data = json.loads(mp.read_text(encoding='utf-8'))
        data['status'] = 'running'; data['cancel'] = False
        atomic_json(mp, data)
        def stopping():
            return (run/'cancel.json').exists()
        try:
            for record in data['records']:
                if stopping():
                    data['cancel'] = True; break
                if record['status'] == 'success' and _verified_record(record):
                    progress('Verified cache: ' + record['source_name']); continue
                try:
                    for item in record['source_stamp']:
                        st=Path(item['path']).stat()
                        if (st.st_size,st.st_mtime_ns)!=(item['size'],item['mtime_ns']):
                            raise ValueError('Input changed since batch snapshot; import it as a new batch')
                    if record['estimated_block_bytes'] > data['memory_budget_mb'] * 1024**2:
                        raise MemoryError('Estimated row working set exceeds configured memory budget')
                    control = run / (record['scan_id'] + '.control.json')
                    atomic_json(control, {'cancel': False})
                    cmd = [sys.executable, '-u', '-m', 'ebsd_autoplot', record['input'],
                           '--output-root', data['output_root'], '--scan', record['scan_path'],
                           '--source-name',record['source_name'],'--scan-tag',record.get('scan_tag',''),
                           *data['options'], '--control-file', str(control)]
                    record.update(status='running', argv=cmd); atomic_json(mp, data)
                    proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parent.parent,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                    monitor_stop=threading.Event()
                    def monitor():
                        while not monitor_stop.wait(.15):
                            if stopping():atomic_json(control, {'cancel':True});return
                    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
                    with (run/(record['scan_id']+'.log')).open('a', encoding='utf-8') as log:
                        for line in proc.stdout:
                            log.write(line); log.flush(); progress(line.rstrip())
                            if stopping(): atomic_json(control, {'cancel': True})
                            if line.startswith('JOB_EVENT '):
                                ev=json.loads(line[len('JOB_EVENT '):])
                                if ev['kind']=='analysis': record['run_path']=ev['path']
                    code=proc.wait()
                    monitor_stop.set();watcher.join(timeout=1)
                    record.update(returncode=code, status='success' if code==0 else ('cancelled' if code==130 else 'failed'))
                    if code: record['error']='Child exit '+str(code)+'; see '+record['scan_id']+'.log'
                except Exception as exc:
                    record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    progress(record['error'])
                atomic_json(mp, data); _reports(run, data)
            states=[r['status'] for r in data['records']]
            data['status']='cancelled' if stopping() else ('failed' if 'failed' in states else 'success')
        except BaseException:
            data['status']='interrupted'; raise
        finally:
            atomic_json(mp, data); _reports(run, data)
    return data


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('run'); args=p.parse_args()
    raise SystemExit(0 if execute_batch(args.run)['status']=='success' else 1)
