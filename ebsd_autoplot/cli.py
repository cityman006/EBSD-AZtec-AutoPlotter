from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import threading
import traceback
from .archive import preview_legacy, import_legacy
from .discovery import discover_scans
from .io import H5Source
from .compute import prepare
from .v2 import run_v2
from . import pipeline
from .storage import Cancelled, cancellation

def _resolve_cache(value):
    path=Path(value).expanduser().resolve()
    if path.is_dir():path=path/'features.h5'
    if not path.is_file():raise FileNotFoundError(f'Analysis cache not found: {path}')
    return path

def _render_only(cache,output_root=None,style=None,formats=('png',)):
    style=style or {}
    options={'font':style.get('font','DejaVu Sans'),'font_size':style.get('font.size',12),
             'svg_text_mode':style.get('svg_text_mode','editable_text'),'formats':list(formats)}
    return pipeline.render_only(cache,options)

def parser():
    p=argparse.ArgumentParser(description='EBSD AutoPlotter | per-scan archives, atomic artifacts, verified Resume')
    p.add_argument('input',nargs='?',help='Raw H5/H5OINA or montage directory; all discovered scans are isolated automatically')
    p.add_argument('--output-root',help='Explicit total output root')
    p.add_argument('--out',help='Legacy alias for --output-root; never reuses a previous run directory')
    p.add_argument('--scan',action='append',help='Optional scan_path or scan tag; repeat to select several. Default: all scans')
    p.add_argument('--cache',help='Render Only from a selected archived cache; never opens raw data')
    p.add_argument('--analysis-cache',help='V2 from a selected archived V1 cache')
    p.add_argument('--resume',help='Explicitly resume an analysis/render manifest or run directory after verifying checkpoints')
    p.add_argument('--recover-interrupted',metavar='OUTPUT_ROOT',help='Mark abandoned running manifests after checking their owner processes')
    p.add_argument('--control-file',help='Per-job cancellation request file used by the GUI')
    p.add_argument('--preview-legacy',metavar='OLD_OUTPUT')
    p.add_argument('--import-legacy',metavar='OLD_OUTPUT')
    p.add_argument('--include-unassigned-legacy',action='store_true')
    p.add_argument('--font',default='DejaVu Sans')
    p.add_argument('--font-size',type=float,default=12)
    p.add_argument('--formats',nargs='+',choices=['png','tiff','pdf','svg'],default=['png'])
    p.add_argument('--svg-text-mode',choices=['editable_text','text_as_paths'],default='editable_text')
    p.add_argument('--analysis',choices=['v1','v1v2','v2','inspect','render',
        'v3_texture','v3_schmid','v3_variants','v3_prior_beta'],default='v1')
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--roi',nargs=4,type=int,metavar=('X','Y','W','H'))
    p.add_argument('--grain-threshold',type=float,default=5)
    p.add_argument('--min-grain-pixels',type=int,default=8)
    p.add_argument('--no-ipf',action='store_true',help='Explicit unsupported-environment fallback; creates no fake IPF')
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--texture',action='store_true')
    p.add_argument('--v3-weighting',choices=['area','grain'],default='area')
    p.add_argument('--v3-bandwidth',type=float,default=10.0)
    p.add_argument('--v3-grid-step',type=float,default=15.0)
    p.add_argument('--v3-max-samples',type=int,default=2000)
    p.add_argument('--v3-seed',type=int,default=11)
    p.add_argument('--v3-load-axis',nargs=3,type=float,default=[0.,0.,1.],metavar=('X','Y','Z'))
    p.add_argument('--mtex-roi',nargs=4,type=int,metavar=('X','Y','W','H'))
    return p

def main(argv=None):
    p=parser();args=p.parse_args(argv);opts=vars(args).copy()
    stop=threading.Event();old_handlers={}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT,signal.SIGTERM):
            old_handlers[sig]=signal.signal(sig,lambda *_:stop.set())
    def cancelled():
        if stop.is_set():return True
        if args.control_file and Path(args.control_file).exists():
            try:return bool(json.loads(Path(args.control_file).read_text(encoding='utf-8')).get('cancel'))
            except (OSError,ValueError):return True
        return False
    try:
        with cancellation(cancelled):
            if args.recover_interrupted:
                print(json.dumps(pipeline.recover_interrupted(args.recover_interrupted),ensure_ascii=False,indent=2));return 0
            if args.resume:
                print('Resumed:',pipeline.resume(args.resume));return 0
            if args.preview_legacy:
                print(json.dumps(preview_legacy(args.preview_legacy),ensure_ascii=False,indent=2));return 0
            if args.import_legacy:
                if not args.output_root and not args.out:p.error('--import-legacy requires --output-root')
                print(json.dumps([str(x) for x in import_legacy(args.import_legacy,args.output_root or args.out,
                    include_unassigned=args.include_unassigned_legacy)],ensure_ascii=False,indent=2));return 0
            if args.cache:
                target=pipeline.render_only(_resolve_cache(args.cache),opts)
                print('Rendered from cache:',args.cache,'->',target);return 0
            if args.analysis=='v2':
                if not args.analysis_cache:p.error('V2 requires --analysis-cache')
                print('V2 results:',pipeline.v2_from_cache(_resolve_cache(args.analysis_cache),opts));return 0
            if not args.input:p.error('input is required for Analyze')
            if args.analysis=='render':p.error('Render Only requires --cache')
            if args.batch<1:p.error('--batch must be positive')
            specs=discover_scans(args.input)
            if args.scan:
                specs=[s for s in specs if s.scan_path in args.scan or s.scan_tag in args.scan]
                if not specs:p.error('No scan matches --scan')
            if args.analysis=='inspect':
                for spec in specs:print(json.dumps(H5Source(spec).header,ensure_ascii=False,indent=2))
                return 0
            if not args.output_root and not args.out:p.error('specify --output-root (or legacy --out) explicitly')
            if args.analysis in ('v3_schmid','v3_variants','v3_prior_beta'):
                try:
                    roi=args.roi or [0,0,sources[0].nx,sources[0].ny]
                    if len(roi)!=4 or min(roi[:2])<0 or min(roi[2:])<=0 or roi[0]+roi[2]>sources[0].nx or roi[1]+roi[3]>sources[0].ny:
                        raise ValueError()
                except (ValueError,IndexError):p.error('V3 grain analyses require an ROI within the scan dimensions')
            if args.v3_bandwidth<=0 or args.v3_grid_step<=0 or args.v3_max_samples<1:p.error('V3 bandwidth, grid step and sample count must be positive')
            opts['v3_load_axis']=tuple(args.v3_load_axis)
            failed=False
            sources=[H5Source(spec) for spec in specs]
            for source in sources:
                if cancelled():raise Cancelled('Cancelled before next scan')
                try:pipeline.analyze(source,args.output_root or args.out,opts)
                except Cancelled:raise
                except Exception:traceback.print_exc();failed=True
            return 1 if failed else 0
    except Cancelled as exc:
        print(str(exc));return 130
    except Exception:
        traceback.print_exc();return 1
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)

if __name__=='__main__':raise SystemExit(main())
