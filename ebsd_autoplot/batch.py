"""Persisted, resource-admitted batches with isolated scientific child processes."""
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
import shutil
import queue
import math
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


def _option(argv, name, default=None, count=1):
    argv = list(argv)
    for index, token in enumerate(argv):
        if token == name:
            values = argv[index + 1:index + 1 + count]
            if len(values) != count:raise ValueError(f'Missing value for {name}')
            return values[0] if count == 1 else values
        if str(token).startswith(name + '='):
            if count != 1:raise ValueError(f'{name} requires {count} separate values')
            return str(token).split('=', 1)[1]
    return default


def _resource_estimate(record, argv):
    """Conservative scheduling estimate, not a measurement or OS memory cap."""
    rows = int(_option(argv, '--batch', 32))
    if rows < 1:raise ValueError('Block rows must be positive')
    if 'nx' not in record or 'ny' not in record:
        # Preserve execution of old external manifests with block-only records.
        total = max(1, int(record.get('estimated_block_bytes', 1)))
        return {'model': 'legacy block-only; create a new batch for full resource estimates',
                'baseline_bytes': 0, 'block_bytes': total, 'grain_roi_bytes': 0,
                'render_bytes': 0, 'total_bytes': total, 'hard_os_limit': False}
    nx, ny = int(record['nx']), int(record['ny'])
    pixels = nx * ny
    mode = _option(argv, '--analysis', 'v1')
    roi = _option(argv, '--roi', None, count=4)
    grain_pixels = pixels
    if roi is not None:
        x, y, w, h = map(int, roi)
        if min(x, y) < 0 or min(w, h) < 1 or x + w > nx or y + h > ny:
            raise ValueError('Batch ROI lies outside a selected scan')
        grain_pixels = w * h
    grain_mode = (mode in ('v1v2', 'v2', 'v3_schmid', 'v3_variants', 'v3_prior_beta', 'v4_quantitative') or
                  (mode == 'v3_texture' and _option(argv, '--v3-weighting', 'area') == 'grain'))
    baseline = 256 * 1024**2
    block = nx * min(rows, ny) * 160
    grain = grain_pixels * (160 if mode in ('v3_variants', 'v3_prior_beta') else 128) if grain_mode else 0
    # Native panels retain raster data; this is full field even for a V2 ROI.
    render = pixels * 48 if '--no-render' not in argv else 0
    total = baseline + block + max(grain, render)
    return {'model': 'baseline + row working set + largest full-map/ROI stage',
            'baseline_bytes': baseline, 'block_bytes': block, 'grain_roi_bytes': grain,
            'render_bytes': render, 'grain_roi_pixels': grain_pixels if grain_mode else 0,
            'total_bytes': total, 'hard_os_limit': False,
            'limitation': 'Admission estimate only; actual Python/library/native export memory can exceed this estimate'}


def create_batch(paths, output_root, argv=(), *, memory_budget_mb=2048, concurrency=1):
    if not math.isfinite(memory_budget_mb) or memory_budget_mb < 16:
        raise ValueError('Memory budget must be at least 16 MB')
    if isinstance(concurrency, bool) or int(concurrency) != concurrency or not 1 <= int(concurrency) <= 8:
        raise ValueError('Concurrency must be an integer from 1 to 8')
    concurrency = int(concurrency)
    # Disallow arguments that could redirect a frozen batch snapshot.
    reserved = {'--output-root', '--out', '--scan', '--source-name', '--scan-tag', '--control-file', '--cache', '--resume'}
    if any(str(x).split('=')[0].startswith('--') and
           any(option.startswith(str(x).split('=')[0]) for option in reserved) for x in argv):
        raise ValueError('Batch options cannot override input/output/scan identity')
    records = inventory(paths)
    for record in records:
        estimate = _resource_estimate(record, argv)
        record.update(resource_estimate=estimate, estimated_total_bytes=estimate['total_bytes'],
                      estimated_block_bytes=estimate['block_bytes'])
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    estimate=sum(r['estimated_cache_bytes'] for r in records)*(2 if '--no-render' in argv else 5)
    if shutil.disk_usage(root).free<estimate:
        raise OSError('Insufficient free disk space for the estimated caches and exports; choose another output root')
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
    data = {'schema_version': 2, 'batch_id': run.name, 'status': 'queued',
            'output_root': str(root), 'options': list(argv), 'concurrency': concurrency,
            'resource_policy': 'Sum of all active child estimates admitted within memory budget; not an OS hard memory cap',
            'threads_per_child': 1,
            'memory_budget_mb': memory_budget_mb, 'records': records, 'cancel': False}
    data['estimated_disk_bytes']=estimate
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
    numeric=[]
    for r in data['records']:
        if r['status']!='success' or not r.get('run_path'):continue
        p=Path(r['run_path'])/'analysis_cache/v4_quantitative/results/sample_phase_summary.csv'
        if p.is_file():
            with p.open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    row.update(source_name=r['source_name'],sample_id=r['scan_id'],source_run_id=Path(r['run_path']).name)
                    numeric.append(row)
    if numeric:
        temp=run/'.Summary.csv.tmp'
        fields=list(dict.fromkeys(k for r in numeric for k in r))
        with temp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(numeric)
        os.replace(temp,run/'Summary.csv')


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
        operations['v4_quantitative']=(cache/'v4_quantitative/results','ebsd_autoplot.quantitative.analyze_quantitative')
        for stage in m.get('checkpoints', {}):
            if stage in operations and not _receipt(*operations[stage]): return False
        if not m['options'].get('no_render', False):
            from .pipeline import _dataset_for
            if not validate_receipt(_dataset_for(mp.parent)/m['render_path'], 'ebsd_autoplot.render.maps'): return False
        return True
    except (KeyError, ValueError, OSError):
        return False


def _stop_owned_child(proc, control, timeout=3):
    """Do not leave an owned analysis running after its log consumer failed.

    Normal cancellation remains cooperative. An exception in the supervising
    process first requests the same safe checkpoint cancellation, then bounds
    cleanup of this exact child so a subsequent sample cannot overlap it.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        if control is not None:atomic_json(control, {'cancel':True})
    except OSError:
        pass  # Still stop the owned child if its control location is unavailable.
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait(timeout=timeout)


def execute_batch(run, *, progress=print):
    """Coordinate bounded isolated children; only this thread mutates records.

    Reader threads perform pipe I/O only and publish events into a bounded
    queue. All controls, log files, manifest snapshots and reports belong to
    the coordinator. Memory is estimated admission, not an OS-enforced cap.
    """
    run = Path(run).resolve(); mp = run / 'batch_manifest.json'
    with file_lock(run / '.batch.lock', timeout=.1):
        data = json.loads(mp.read_text(encoding='utf-8'))
        # An abruptly killed supervisor can leave its child alive. Never
        # duplicate that analysis or terminate an unowned process by PID.
        for record in data['records']:
            if record.get('status') == 'running' and record.get('child_pid'):
                from .pipeline import process_token
                token = process_token(record['child_pid'])
                if token in ('alive', 'access-denied') or (token is not None and
                        (record.get('child_token') is None or token == record['child_token'])):
                    raise ValueError('A child from the previous batch supervisor is still running; wait for it or request cancellation through its saved control file')
        limit = int(data.get('concurrency', 1))
        if not 1 <= limit <= 8:raise ValueError('Manifest concurrency must be from 1 to 8')
        budget = float(data['memory_budget_mb']) * 1024**2
        if not math.isfinite(budget) or not budget > 0:raise ValueError('Manifest memory budget must be positive and finite')
        data.update(status='running', cancel=False)
        events = queue.Queue(maxsize=512)
        active = {}; pending = []

        def stopping():
            return (run / 'cancel.json').exists()

        def snapshot():
            data['resource_state'] = {'active_children': len(active),
                'reserved_bytes': sum(job['reservation'] for job in active.values()),
                'budget_bytes': int(budget), 'active_scan_ids': [job['record']['scan_id'] for job in active.values()],
                'hard_os_limit': False}
            atomic_json(mp, data); _reports(run, data)

        def emit(index, kind, value, stop_event):
            while not stop_event.is_set():
                try:events.put((index, kind, value), timeout=.1); return
                except queue.Full:continue

        def read_pipe(index, proc, stop_event):
            try:
                for line in proc.stdout:
                    if stop_event.is_set():break
                    emit(index, 'line', line, stop_event)
            except Exception as exc:
                emit(index, 'reader_error', f'{type(exc).__name__}: {exc}', stop_event)
            finally:
                emit(index, 'eof', None, stop_event)

        def retire(index, *, error=None, interrupted=False):
            job = active[index]; record = job['record']; proc = job['proc']
            if error is not None or interrupted:
                # Stop enqueueing before bounded cleanup so a broken consumer
                # cannot deadlock a reader on the full event queue.
                job['stop'].set()
                _stop_owned_child(proc, job['control'])
            code = proc.wait()
            job['stop'].set(); job['reader'].join(timeout=2)
            if proc.stdout is not None:proc.stdout.close()
            job['log'].flush(); job['log'].close()
            record['returncode'] = code
            if interrupted:
                record.update(status='interrupted', error=str(error or 'Batch supervisor interrupted'))
            elif error is not None:
                record.update(status='failed', error=str(error))
            else:
                record['status'] = 'success' if code == 0 else 'cancelled' if code == 130 else 'failed'
                if code:record['error'] = f"Child exit {code}; see {record['scan_id']}.log"
                else:record.pop('error', None)
            active.pop(index)

        def launch(index):
            record = data['records'][index]
            for item in record['source_stamp']:
                stat = Path(item['path']).stat()
                if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
                    raise ValueError('Input changed since batch snapshot; import it as a new batch')
            control = run / (record['scan_id'] + '.control.json')
            atomic_json(control, {'cancel': False})
            cmd = [sys.executable, '-u', '-m', 'ebsd_autoplot', record['input'],
                '--output-root', data['output_root'], '--scan', record['scan_path'],
                '--source-name', record['source_name'], '--scan-tag', record.get('scan_tag', ''),
                *data['options'], '--control-file', str(control)]
            env = os.environ.copy()
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
                env[name] = '1'
            record.update(status='running', argv=cmd, child_threads=1)
            record.pop('error', None)
            log = (run / (record['scan_id'] + '.log')).open('a', encoding='utf-8')
            proc = None
            try:
                proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parent.parent,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env)
                stop_event = threading.Event()
                reader = threading.Thread(target=read_pipe, args=(index, proc, stop_event),
                                          name='ebsd-batch-reader-' + record['scan_id'], daemon=True)
                active[index] = {'record': record, 'proc': proc, 'control': control, 'log': log,
                    'reservation': int(record['estimated_total_bytes']), 'stop': stop_event, 'reader': reader,
                    'eof': False, 'cancel_sent': False}
                record['child_pid'] = proc.pid
                from .pipeline import process_token
                record['child_token'] = process_token(proc.pid)
                reader.start()
            except BaseException:
                _stop_owned_child(proc, control)
                if proc is not None and proc.stdout is not None:proc.stdout.close()
                log.close(); active.pop(index, None); raise

        try:
            for index, record in enumerate(data['records']):
                if record['status'] == 'success' and _verified_record(record):
                    progress('Verified cache: ' + record['source_name']); continue
                if 'estimated_total_bytes' not in record:
                    estimate = _resource_estimate(record, data['options'])
                    record.update(resource_estimate=estimate, estimated_total_bytes=estimate['total_bytes'])
                if int(record['estimated_total_bytes']) < 1:raise ValueError('Invalid child memory reservation')
                record['status'] = 'queued';pending.append(index)
            snapshot()
            while pending or active:
                cancelled = stopping()
                if cancelled:
                    data['cancel'] = True
                    # Send to every active child, including silent workers.
                    for job in active.values():
                        if not job['cancel_sent']:
                            atomic_json(job['control'], {'cancel': True}); job['cancel_sent'] = True
                if not cancelled:
                    for index in list(pending):
                        if len(active) >= limit:break
                        record = data['records'][index]; reservation = int(record['estimated_total_bytes'])
                        if reservation > budget:
                            record.update(status='failed', error='Estimated total child memory exceeds batch admission budget; reduce ROI/block rows or increase budget')
                            pending.remove(index);snapshot();continue
                        reserved = sum(job['reservation'] for job in active.values())
                        if reserved + reservation > budget:continue
                        pending.remove(index)
                        try:launch(index)
                        except Exception as exc:
                            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                            try:progress(record['error'])
                            except Exception:pass
                        snapshot()
                if cancelled and not active:break
                try:
                    index, kind, value = events.get(timeout=.05)
                except queue.Empty:
                    index = None
                if index in active:
                    job = active[index]
                    try:
                        if kind == 'line':
                            job['log'].write(value);job['log'].flush()
                            if value.startswith('JOB_EVENT '):
                                event = json.loads(value[len('JOB_EVENT '):])
                                if event['kind'] == 'analysis':job['record']['run_path'] = event['path']
                            record = job['record']
                            progress(f"[{record['source_name']} | {record['scan_id']}] {value.rstrip()}")
                        elif kind == 'reader_error':raise OSError(value)
                        elif kind == 'eof':job['eof'] = True
                    except Exception as exc:
                        retire(index, error=f'{type(exc).__name__}: {exc}');snapshot()
                for index, job in list(active.items()):
                    if job['eof'] and job['proc'].poll() is not None:
                        retire(index);snapshot()
            states = [record['status'] for record in data['records']]
            data['status'] = ('cancelled' if stopping() or 'cancelled' in states else
                              'interrupted' if 'interrupted' in states else
                              'failed' if any(state != 'success' for state in states) else 'success')
        except BaseException as exc:
            data['status'] = 'interrupted'
            # Signal all owned children before waiting for any one of them.
            for job in active.values():
                try:atomic_json(job['control'], {'cancel': True})
                except OSError:pass
            for index in list(active):
                retire(index, error=f'{type(exc).__name__}: {exc}', interrupted=True)
            raise
        finally:
            snapshot()
    return data


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('run'); args=p.parse_args(argv)
    try:status=execute_batch(args.run)['status']
    except KeyboardInterrupt:
        print('Batch interrupted; committed checkpoints are retained',file=sys.stderr)
        return 1
    return 0 if status=='success' else 130 if status=='cancelled' else 1


if __name__ == '__main__':
    raise SystemExit(main())
