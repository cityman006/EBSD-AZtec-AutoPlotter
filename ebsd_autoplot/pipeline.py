"""Manifest-driven jobs with immutable paths, verified stage recovery and cancellation."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
import json
import os
import socket
import sys
import traceback

from .archive import (OutputContext, _new_dir, _source_info, create_context, locate_cache,
                      update_latest, write_run_manifest)
from .compute import prepare
from .discovery import discover_scans
from .io import H5Source
from .mtex import export_roi_ctf
from .render import maps
from .storage import (Cancelled, atomic_json, cancellation, check_cancelled, digest, file_lock,
                      recover_transactions, validate_receipt, _publish, long_path)
from .texture import texture_preview
from .v2 import run_v2
from .texture_v3 import analyze_texture
from .schmid import analyze_schmid
from .alpha_variants import analyze_alpha_variants
from .prior_beta import reconstruct_prior_beta

def now(): return datetime.now().astimezone().isoformat()

def process_token(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            if ctypes.get_last_error()==5:return 'access-denied'
            return None
        try:
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value!=259:return None
            fields=[wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle,*[ctypes.byref(x) for x in fields]):return 'alive'
            return str((fields[0].dwHighDateTime<<32)|fields[0].dwLowDateTime)
        finally:kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
        return Path(f'/proc/{pid}/stat').read_text().split()[21] if Path('/proc').exists() else 'alive'
    except (OSError,IndexError):return None

def owner():return {'pid':os.getpid(),'host':socket.gethostname(),'token':process_token(os.getpid())}

def _dataset_for(run):
    return next(p for p in long_path(run).parents if (p/'dataset_manifest.json').is_file())

def _context(run, manifest):
    run=long_path(run).resolve();dataset=_dataset_for(run); scope=run.parent.parent
    render=dataset/manifest['render_path'] if manifest.get('render_path') else scope/'renders'/manifest.get('render_id','unused')
    if dataset not in render.resolve().parents or render.parent!=scope/'renders':raise ValueError('Render path escaped its scan scope')
    return OutputContext(manifest['dataset_id'],dataset,run.name,run,run/'analysis_cache',render.name,render,
        manifest.get('source_version',''),scope,manifest.get('scan_id',''))

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:stream.flush()

@contextmanager
def job_log(run, manifest):
    logs=Path(run)/'logs';logs.mkdir(exist_ok=True)
    import uuid
    name='run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8]+'.log'
    partial=logs/('.'+name+'.partial')
    try:
        with partial.open('x',encoding='utf-8') as f:
            try:
                with redirect_stdout(Tee(sys.stdout,f)),redirect_stderr(Tee(sys.stderr,f)):yield
            finally:f.flush();os.fsync(f.fileno())
    finally:
        mp=Path(run)/('render_manifest.json' if manifest['kind']=='render' else 'analysis_manifest.json')
        try:
            _publish(partial,logs/name)
            manifest.setdefault('logs',[]).append('logs/'+name)
        except BaseException as exc:
            _failed(manifest,exc);write_run_manifest(mp,manifest);raise
        write_run_manifest(mp,manifest)

def _success(ctx,m,mp,kind,run):
    m.update(status='success',completed_at=now());write_run_manifest(mp,m)
    try:update_latest(ctx,kind,run.name,run)
    except BaseException as exc:
        _failed(m,exc);write_run_manifest(mp,m);raise

def event(kind, path, **extra):
    print('JOB_EVENT '+json.dumps({'kind':kind,'path':str(path),**extra},ensure_ascii=False),flush=True)

def _mark(mp, m, stage):
    m.setdefault('checkpoints',{})[stage]={'completed_at':now()}
    write_run_manifest(mp,m)
    event('checkpoint',mp.parent,stage=stage)
    check_cancelled()

def _failed(m, exc):
    science=set(m.get('checkpoints',{}))&{'v1','v2','maps','render','texture','mtex','v3_texture','v3_schmid','v3_variants','v3_prior_beta'}
    m.update(status='cancelled' if isinstance(exc,Cancelled) else ('partial' if science else 'failed'),
        error=f'{type(exc).__name__}: {exc}',completed_at=now())

def _source_for(m):
    specs=discover_scans(m['input_path'])
    specs=[s for s in specs if s.scan_path==m['scan_path'] and s.scan_tag==m.get('scan_tag','') and s.source_name==m['source_name']]
    if len(specs)!=1:raise ValueError('Source scan identity changed; start a new Analyze instead of Resume')
    src=H5Source(specs[0])
    info=_source_info(src)
    if info['source_version']!=m['source_version'] or info['dataset_id']!=m.get('source_identity',m['dataset_id']):
        raise ValueError('Source version changed; refusing to resume an old analysis')
    return src

def _receipt(root, operation):
    recover_transactions(root)
    return validate_receipt(root,operation)

def validate_cache(cache,analysis):
    cache=long_path(cache)
    if validate_receipt(cache.parent,'ebsd_autoplot.compute.prepare'):return
    if analysis.get('schema_version',1)>=2 and not analysis.get('legacy_import'):
        raise ValueError('Cache has no complete verified V1 receipt')
    if analysis.get('status')!='success':raise ValueError('Legacy cache does not belong to a successful analysis')
    # Older caches have no transaction journals. Validate their stored schema
    # read-only, then pin a complete hash in the *new* render/V2 manifest.
    import h5py
    with h5py.File(cache,'r') as f:
        if not {'features','indexed'}<=set(f):raise ValueError('Legacy cache fields are incomplete')
        shape=f['indexed'].shape
        if len(shape)!=2 or f['features'].shape!=(7,*shape):raise ValueError('Legacy cache grid is inconsistent')
        json.loads(f.attrs['metadata_json'])
        for key in ('mad','pq','kam_valid'):
            if key in f and f[key].shape!=shape:raise ValueError(f'Legacy field has the wrong shape: {key}')
        for key in ('ipf_X','ipf_Y','ipf_Z'):
            if key in f and f[key].shape!=(*shape,3):raise ValueError('Legacy IPF grid is inconsistent')

def analyze(source, output_root, options):
    ctx=create_context(source,output_root)
    m={'schema_version':2,'kind':'analysis','dataset_id':ctx.dataset_id,'scan_id':ctx.scan_id,
        'analysis_run_id':ctx.analysis_run_id,'cache_id':ctx.analysis_run_id,'source_version':ctx.source_version,
        'source_identity':_source_info(source)['dataset_id'],
        'source':source.header,'source_name':source.spec.source_name,'input_path':str(source.path),
        'scan_path':source.scan_path,'scan_tag':source.spec.scan_tag,'options':options,
        'render_path':ctx.render_dir.relative_to(ctx.dataset_dir).as_posix(),
        'cache_path':(ctx.cache_dir/'features.h5').relative_to(ctx.dataset_dir).as_posix(),
        'status':'running','owner':owner(),'started_at':now(),'checkpoints':{'source_validated':{'source_version':ctx.source_version}}}
    write_run_manifest(ctx.analysis_dir/'analysis_manifest.json',m)
    return execute_analysis(ctx.analysis_dir,m,source)

def execute_analysis(run,m,source=None):
    run=long_path(run).resolve();mp=run/'analysis_manifest.json';ctx=_context(run,m);opts=m['options']
    with file_lock(run/'.run.lock',timeout=.1):
        m.update(status='running',owner=owner(),resumed_at=now());write_run_manifest(mp,m)
        event('analysis',run,scan_id=m['scan_id'],dataset_dir=str(ctx.dataset_dir))
        with job_log(run,m):
            try:
                check_cancelled()
                complete=_receipt(ctx.cache_dir,'ebsd_autoplot.compute.prepare')
                if complete and 'v1' not in m.get('checkpoints',{}):source=source or _source_for(m)
                if not complete:
                    if m.get('checkpoints',{}).get('v1'):raise ValueError('V1 receipt missing; checkpoint cannot be trusted')
                    source=source or _source_for(m)
                    prepare(source,ctx.cache_dir,batch=opts['batch'],do_ipf=not opts['no_ipf'])
                if 'v1' not in m.get('checkpoints',{}):
                    if _source_info(source)['source_version']!=m['source_version']:
                        raise ValueError('Input changed during analysis; start a new Analyze')
                metadata=ctx.cache_dir/'source_metadata.json'
                if metadata.exists() and json.loads(metadata.read_text(encoding='utf-8'))!=m['source']:
                    raise ValueError('Source metadata snapshot was modified')
                if not metadata.exists():atomic_json(metadata,m['source'])
                _mark(mp,m,'v1')
                fp=ctx.cache_dir/'features.h5'
                analysis_mode=opts['analysis']
                needs_v2=(analysis_mode in ('v1v2','v3_schmid','v3_variants','v3_prior_beta','v4_quantitative') or
                          (analysis_mode=='v3_texture' and opts.get('v3_weighting','area')=='grain'))
                if needs_v2:
                    if not _receipt(ctx.cache_dir/'v2','ebsd_autoplot.v2.run_v2'):
                        run_v2(fp,ctx.cache_dir/'v2',roi=opts['roi'],threshold=opts['grain_threshold'],min_size=opts['min_grain_pixels'])
                    _mark(mp,m,'v2')
                if opts['texture']:
                    if not _receipt(ctx.cache_dir/'texture','ebsd_autoplot.texture.texture_preview'):
                        texture_preview(fp,ctx.cache_dir/'texture')
                    _mark(mp,m,'texture')
                if opts['mtex_roi']:
                    if not _receipt(ctx.cache_dir/'mtex','ebsd_autoplot.mtex.export_roi_ctf'):
                        source=source or _source_for(m)
                        export_roi_ctf(source,ctx.cache_dir/'mtex'/'verification_roi.ctf',*opts['mtex_roi'])
                    _mark(mp,m,'mtex')
                v3_mode=analysis_mode.removeprefix('v3_') if analysis_mode.startswith('v3_') else None
                if v3_mode:
                    v3root=ctx.cache_dir/'v3'/v3_mode/'results'
                    v3root.mkdir(parents=True,exist_ok=True)
                    operations={'texture':'ebsd_autoplot.texture_v3.analyze_texture',
                        'schmid':'ebsd_autoplot.schmid.analyze_schmid',
                        'variants':'ebsd_autoplot.alpha_variants.analyze_alpha_variants',
                        'prior_beta':'ebsd_autoplot.prior_beta.reconstruct_prior_beta'}
                    if not _receipt(v3root,operations[v3_mode]):
                        if v3_mode=='texture':
                            analyze_texture(fp,v3root,weighting=opts.get('v3_weighting','area'),
                                bandwidth_deg=opts.get('v3_bandwidth',10.0),grid_step_deg=opts.get('v3_grid_step',15.0),
                                max_samples=opts.get('v3_max_samples',3000),seed=opts.get('v3_seed',0),formats=opts['formats'])
                        elif v3_mode=='schmid':
                            analyze_schmid(fp,v3root,load_axis=opts.get('v3_load_axis',(0,0,1)),
                                roi=opts.get('roi'),block_rows=opts.get('batch',32),formats=opts['formats'])
                        elif v3_mode=='variants':
                            analyze_alpha_variants(fp,ctx.cache_dir/'v2',v3root,formats=opts['formats'])
                        else:
                            variants=ctx.cache_dir/'v3'/'variants'/'results'
                            if not _receipt(variants,operations['variants']):
                                analyze_alpha_variants(fp,ctx.cache_dir/'v2',variants,formats=opts['formats'])
                            reconstruct_prior_beta(fp,ctx.cache_dir/'v2',variants,v3root,formats=opts['formats'])
                    _mark(mp,m,'v3_'+v3_mode)
                if analysis_mode=='v4_quantitative':
                    from .quantitative import analyze_quantitative
                    target=ctx.cache_dir/'v4_quantitative'/'results'
                    if not _receipt(target,'ebsd_autoplot.quantitative.analyze_quantitative'):
                        analyze_quantitative(fp,ctx.cache_dir/'v2',target,formats=opts['formats'],font=opts['font'],
                            thresholds=opts.get('rx_threshold_config'),min_pixels=opts['min_grain_pixels'])
                    _mark(mp,m,'v4_quantitative')
                if not opts['no_render']:
                    render_only(fp,opts,ctx.render_dir,allow_partial=True)
                    _mark(mp,m,'render')
            except BaseException as exc:
                traceback.print_exc();_failed(m,exc);write_run_manifest(mp,m);raise
        _success(ctx,m,mp,'analysis',run)
    print('Done:',run,flush=True)
    return run

def render_only(cache,options,render_dir=None,allow_partial=False):
    cache=long_path(cache).resolve();dataset,scope,source_run=locate_cache(cache)
    analysis=json.loads((source_run/'analysis_manifest.json').read_text(encoding='utf-8'))
    if not allow_partial and analysis['status'] not in ('success','partial','cancelled'):
        raise ValueError('Selected analysis has no verified V1 result')
    validate_cache(cache,analysis)
    if render_dir is None:_,render_dir=_new_dir(scope/'renders','render')
    render_dir=long_path(render_dir);mp=render_dir/'render_manifest.json'
    if mp.exists():
        m=json.loads(mp.read_text(encoding='utf-8'))
        if m.get('status')=='success':
            if not validate_receipt(render_dir,'ebsd_autoplot.render.maps'):raise ValueError('Completed render was modified')
            return render_dir
        if m.get('cache_path')!=cache.relative_to(dataset).as_posix():raise ValueError('Render cache identity mismatch')
    else:
        m={'schema_version':2,'kind':'render','dataset_id':analysis['dataset_id'],'scan_id':analysis.get('scan_id'),
            'render_id':render_dir.name,'analysis_run_id':source_run.name,'cache_id':source_run.name,
            'cache_path':cache.relative_to(dataset).as_posix(),'cache_sha256':digest(cache),
            'options':options,'style_name':options['font'],'actual_font':options['font'],
            'formats':options['formats'],'started_at':now()}
    return execute_render(render_dir,m)

def execute_render(run,m):
    run=long_path(run).resolve();dataset=_dataset_for(run);scope=run.parent.parent;mp=run/'render_manifest.json'
    cache=(dataset/m['cache_path']).resolve()
    opts=m['options']
    style={'name':opts['font'],'font':opts['font'],'font.size':opts['font_size'],'svg_text_mode':opts['svg_text_mode']}
    style.update(maps=opts.get('maps'),layout=opts.get('layout',(2,4)))
    with file_lock(run/'.run.lock',timeout=.1):
        m.update(status='running',owner=owner());write_run_manifest(mp,m)
        event('render',run,cache_id=m['cache_id'],scan_id=m.get('scan_id'))
        with job_log(run,m):
            try:
                check_cancelled()
                if dataset not in cache.parents or digest(cache)!=m['cache_sha256']:raise ValueError('Selected cache changed since this render started')
                style_path=run/'publication_style.json'
                if style_path.exists() and json.loads(style_path.read_text(encoding='utf-8'))!=style:
                    raise ValueError('Style snapshot was modified; use a new Render Only record')
                if not style_path.exists():atomic_json(style_path,style)
                if not _receipt(run,'ebsd_autoplot.render.maps'):
                    maps(cache,run,style=style,formats=opts['formats'])
                _mark(mp,m,'maps')
            except BaseException as exc:
                traceback.print_exc();_failed(m,exc);write_run_manifest(mp,m);raise
        ctx=OutputContext(m['dataset_id'],dataset,m['analysis_run_id'],cache.parent.parent,cache.parent,
            run.name,run,'',scope,m.get('scan_id',''))
        _success(ctx,m,mp,'render',run)
    return run

def v2_from_cache(cache,options):
    cache=long_path(cache).resolve();dataset,scope,source_run=locate_cache(cache)
    original=json.loads((source_run/'analysis_manifest.json').read_text(encoding='utf-8'))
    validate_cache(cache,original)
    run_id,run=_new_dir(scope/'analysis_runs','analysis')
    m={'schema_version':2,'kind':'v2','dataset_id':original['dataset_id'],'scan_id':original.get('scan_id'),
        'analysis_run_id':run_id,'source_analysis_run_id':source_run.name,'source_version':original.get('source_version'),
        'cache_path':cache.relative_to(dataset).as_posix(),'cache_sha256':digest(cache),'options':options,
        'status':'running','started_at':now()}
    write_run_manifest(run/'analysis_manifest.json',m)
    return execute_v2(run,m)

def execute_v2(run,m):
    run=long_path(run).resolve();dataset=_dataset_for(run);scope=run.parent.parent;mp=run/'analysis_manifest.json'
    cache=(dataset/m['cache_path']).resolve();opts=m['options']
    with file_lock(run/'.run.lock',timeout=.1):
        m.update(status='running',owner=owner());write_run_manifest(mp,m);event('analysis',run)
        with job_log(run,m):
            try:
                check_cancelled()
                if dataset not in cache.parents or digest(cache)!=m['cache_sha256']:raise ValueError('Referenced cache was modified')
                if not _receipt(run/'v2','ebsd_autoplot.v2.run_v2'):
                    run_v2(cache,run/'v2',roi=opts['roi'],threshold=opts['grain_threshold'],min_size=opts['min_grain_pixels'])
                _mark(mp,m,'v2')
            except BaseException as exc:
                traceback.print_exc();_failed(m,exc);write_run_manifest(mp,m);raise
        ctx=OutputContext(m['dataset_id'],dataset,run.name,run,run/'analysis_cache','',scope/'renders'/'unused',m.get('source_version',''),scope,m.get('scan_id',''))
        _success(ctx,m,mp,'analysis',run)
    return run

def recover_interrupted(root):
    recovered=[]
    manifests=sorted({*Path(root).rglob('analysis_manifest.json'),*Path(root).rglob('render_manifest.json')})
    for mp in manifests:
        m=json.loads(mp.read_text(encoding='utf-8'))
        if m.get('status')!='running':continue
        who=m.get('owner',{})
        if who.get('host') not in (None,socket.gethostname()):continue
        token=process_token(who['pid']) if who.get('pid') else None
        if token in ('access-denied','alive') or (token is not None and token==who.get('token')):continue
        try:
            with file_lock(mp.parent/'.run.lock',timeout=.01):
                science=set(m.get('checkpoints',{}))&{'v1','v2','maps','render','texture','mtex','v3_texture','v3_schmid','v3_variants','v3_prior_beta'}
                m.update(status='partial' if science else 'failed',interrupted=True,
                    error='Owner process ended before task completion',completed_at=now())
                write_run_manifest(mp,m);recovered.append(str(mp))
        except TimeoutError:continue
    return recovered

def resume(path):
    path=long_path(path).resolve()
    if path.is_dir():
        path=path/('render_manifest.json' if path.parent.name=='renders' else 'analysis_manifest.json')
    m=json.loads(path.read_text(encoding='utf-8'))
    if m.get('schema_version',0)<2:raise ValueError('This older run has no validated recovery snapshot; create a new run')
    if m.get('status')=='success':raise ValueError('A successful historical task cannot be resumed')
    who=m.get('owner',{})
    token=process_token(who['pid']) if who.get('pid') else None
    if m.get('status')=='running' and (token in ('access-denied','alive') or (token is not None and token==who.get('token'))):
        raise ValueError('Task is still running or its owner cannot be verified')
    m.setdefault('resume_events',[]).append({'at':now(),'previous_status':m['status'],
        'previous_error':m.get('error'),'interrupted':m.get('interrupted',False)})
    for key in ('error','interrupted','completed_at'):m.pop(key,None)
    return {'analysis':execute_analysis,'render':execute_render,'v2':execute_v2}[m['kind']](path.parent,m)
