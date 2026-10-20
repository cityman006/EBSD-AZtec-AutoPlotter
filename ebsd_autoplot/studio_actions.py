"""Isolated cached scientific actions; every invocation gets an independent run."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid
from .storage import atomic_json,cancellation,Cancelled


def run_action(action,cache,output_root,*,config=None,control_file=None):
    config=json.loads(json.dumps(config or {}))
    run=Path(output_root)/'studio_runs'/(action+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    manifest={'schema_version':1,'action':action,'status':'running','source_cache':str(cache),
              'config':json.loads(json.dumps(config)),'scientific_status':'research; independent validation pending'}
    def cancelled():
        if not control_file:return False
        try:return bool(json.loads(Path(control_file).read_text(encoding='utf-8')).get('cancel'))
        except (OSError,ValueError):return True
    atomic_json(run/'studio_manifest.json',manifest)
    try:
        with cancellation(cancelled):
            if action in ('quantitative','patches','graph','eds'):
                from .archive import locate_cache
                from .pipeline import validate_cache
                _,_,source_run=locate_cache(cache)
                source_manifest=json.loads((source_run/'analysis_manifest.json').read_text(encoding='utf-8'))
                validate_cache(cache,source_manifest)
            if action=='quantitative':
                from .quantitative import analyze_quantitative
                from .v2 import run_v2
                v2=Path(cache).parent/'v2'
                if not (v2/'v2_grains.h5').is_file():
                    v2=run/'v2';run_v2(cache,v2,roi=config.pop('roi',None))
                else:
                    from .storage import validate_receipt
                    if not validate_receipt(v2,'ebsd_autoplot.v2.run_v2'):
                        raise ValueError('V2 cache lacks a verified transaction; create a new V2 result from the selected V1 cache')
                analyze_quantitative(cache,v2,run/'results',**config)
            elif action in ('patches','graph'):
                v2=Path(cache).parent/'v2'
                if action=='graph':
                    from .graph_builder import build_graph
                    if not (v2/'v2_grains.h5').is_file():
                        from .v2 import run_v2
                        v2=run/'v2';run_v2(cache,v2,roi=config.pop('roi',None))
                    build_graph(cache,v2,run/'results',**config)
                else:
                    from .slicing import export_patches
                    if config.pop('include_eds',False):
                        from .eds import load_h5oina_eds
                        from .pipeline import _source_for
                        source=_source_for(source_manifest)
                        if len(source.parts)!=1:raise ValueError('EDS montage stitching requires explicit registration; export pure EBSD patches instead')
                        config['eds']=load_h5oina_eds(source.path,source.scan_path)
                    if config.pop('include_grains',False):config['v2_dir']=v2
                    export_patches(cache,run/'results',**config)
            elif action=='eds':
                from .eds import load_h5oina_eds,save_eds
                from .pipeline import _source_for
                import h5py
                source=_source_for(source_manifest)
                if len(source.parts)!=1:raise ValueError('EDS montage stitching is not implemented; use an explicitly registered single source')
                result=load_h5oina_eds(source.path,source.scan_path,**config)
                with h5py.File(cache,'r') as f:
                    phase=f['features'][4,:,:];indexed=f['indexed'][:]>0
                save_eds(result,run/'results',phase_map=phase,indexed=indexed,
                         phase_names={i:s['name'] for i,s in source.phase_specs.items()})
                print('EDS: '+result.metadata.get('status','available' if result.available else 'unavailable'),flush=True)
            else:raise ValueError('Unknown cached action: '+action)
        manifest['status']='success'
    except BaseException as exc:
        manifest.update(status='cancelled' if isinstance(exc,Cancelled) else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:atomic_json(run/'studio_manifest.json',manifest)
    print('Studio results: '+str(run),flush=True)
    return run


def main():
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--cache');p.add_argument('--output-root',required=True)
    p.add_argument('--config',default='{}');p.add_argument('--control-file');a=p.parse_args()
    try:run_action(a.action,a.cache,a.output_root,config=json.loads(a.config),control_file=a.control_file)
    except Cancelled:return 130
    return 0

if __name__=='__main__':raise SystemExit(main())
