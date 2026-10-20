"""Publication panels and full-resolution map export without giant Matplotlib figures."""
from __future__ import annotations
from pathlib import Path
import shutil
import time
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from matplotlib.cm import ScalarMappable
from .mad_units import normalise_mad_unit, mad_degrees
from .storage import transactional_output, check_cancelled


@transactional_output()
def maps(features_file,outdir, dpi=220, progress=print, style=None, formats=('png',)):
    """Render cached numeric fields; PDF/SVG keep map images raster and annotations vector."""
    style = dict(style or {})
    formats = tuple(dict.fromkeys(str(x).lower().lstrip('.') for x in formats))
    if not formats:raise ValueError('Choose at least one output format')
    invalid = set(formats) - {'png', 'tiff', 'tif', 'pdf', 'svg'}
    if invalid: raise ValueError(f'Unsupported render format(s): {sorted(invalid)}')
    font = style.get('font', 'DejaVu Sans')
    from matplotlib import font_manager
    try:
        font_manager.findfont(font, fallback_to_default=False)
    except ValueError as exc:
        raise ValueError(f'Requested font is not installed: {font}') from exc
    import matplotlib as mpl
    rc = {'font.family': font, 'svg.fonttype': 'none' if style.get('svg_text_mode', 'editable_text') == 'editable_text' else 'path'}
    for key in ('font.size', 'axes.titlesize', 'axes.labelsize', 'legend.fontsize'):
        if key in style: rc[key] = float(style[key])
    if style.get('bold') is not None: rc['font.weight'] = 'bold' if style['bold'] else 'normal'
    outdir=Path(outdir);singles=outdir/'single_maps';singles.mkdir(parents=True,exist_ok=True)
    native=outdir/'native_maps';native.mkdir(parents=True,exist_ok=True)
    combined=outdir/'combined';combined.mkdir(parents=True,exist_ok=True)
    legends=outdir/'legends';legends.mkdir(parents=True,exist_ok=True)
    (outdir/'preview').mkdir(exist_ok=True)
    with mpl.rc_context(rc=rc), h5py.File(features_file,'r') as f:
        h,w=f['indexed'].shape
        meta=__import__('json').loads(f.attrs['metadata_json'])
        sx=meta['x_step_um'];sy=meta['y_step_um']
        mad_unit=None
        if 'mad' in f:
            if 'Unit' not in f['mad'].attrs:
                raise ValueError('MAD unit missing in features.h5. Regenerate V1 from the verified source in a new output directory.')
            mad_unit=normalise_mad_unit(f['mad'].attrs['Unit'])
        # Quality maps obey raw values; KAM color limits fixed by configuration, never arbitrary channel normalization
        config={
            'KAM':('features',5,'jet',0,5),
            'BC':('features',6,'gray',0,255),
            'MAD':('mad',None,'magma',0,2.0),
            'PQ':('pq',None,'gray',None,None)}
        phase_colors={int(k):v['rgb'] for k,v in meta['phase_specs'].items()}
        all_maps=['IPF_X','IPF_Y','IPF_Z','Phase','KAM','BC','MAD','PQ']
        requested=style.get('maps') or all_maps
        if set(requested)-set(all_maps):raise ValueError('Unknown map selection')
        timings={}
        for name in requested:
            check_cancelled()
            started=time.perf_counter()
            if name.startswith('IPF') and ('ipf_'+name[-1] not in f):continue
            if name in ('MAD','PQ') and name.lower() not in f:continue
            progress(f'Rendering {name}')
            img=np.zeros((h,w,3),dtype=np.uint8)
            if name.startswith('IPF'):
                ds=f['ipf_'+name[-1]]
                for a in range(0,h,128):img[a:a+128]=ds[a:a+128]
            elif name=='Phase':
                ds=f['features']
                for a in range(0,h,128):
                    ph=ds[4,a:a+128].astype(np.uint8)
                    out=np.zeros((*ph.shape,3),np.uint8)
                    for pid,c in phase_colors.items():out[ph==pid]=c
                    img[a:a+128]=out
            else:
                field,chan,cmap,vmin,vmax=config[name]
                ds=f[field] if chan is None else f[field]
                clut=np.rint(255*colormaps[cmap](np.linspace(0,1,256))[:,:3]).astype('u1')
                if vmax is None:
                    # use sampled values for PQ colour scaling; visual only
                    sample=np.asarray(ds[::max(1,h//256),::max(1,w//256)])
                    vmax=float(np.nanpercentile(sample,99))
                    vmin=float(np.nanpercentile(sample,1))
                for a in range(0,h,128):
                    raw=ds[a:a+128] if chan is None else ds[chan,a:a+128,:]
                    if name=='MAD':
                        raw=mad_degrees(raw,mad_unit)
                    # Invalid numeric fields never become a real low/high colour.
                    good=np.isfinite(raw)
                    if name=='MAD':good&=f['indexed'][a:a+128,:].astype(bool)&(raw>=0)
                    if field=='features':
                        good&=f['indexed'][a:a+128,:].astype(bool)
                        if name=='KAM':good&=f['kam_valid'][a:a+128,:].astype(bool)
                    norm=np.clip((np.where(good,raw,vmin)-vmin)/max(vmax-vmin,1e-9),0,1)
                    pix=clut[(norm*255).astype('u1')]
                    pix[~good]=0
                    img[a:a+128]=pix
            image=Image.fromarray(img,'RGB')
            image.save(singles/(name+'.png'))
            shutil.copyfile(singles/(name+'.png'),native/(name+'.png'))
            if 'tiff' in formats or 'tif' in formats:
                image.save(native/(name+'.tiff'))
            # interpolation='none' preserves native raster pixels in PDF/SVG;
            # labels and scale bars remain vector paths/text.
            if {'pdf','svg'} & set(formats):
                single,axis=plt.subplots(figsize=(8,8*max(h/w,.1)),layout='constrained')
                axis.imshow(img,interpolation='none',extent=(0,w*sx,h*sy,0))
                axis.set_xlabel('X (µm)');axis.set_ylabel('Y (µm)');axis.set_title(name.replace('_','-'))
                for fmt in ('pdf','svg'):
                    if fmt in formats:single.savefig(singles/(name+'.'+fmt),dpi=dpi)
                plt.close(single)
            timings[name]=time.perf_counter()-started
            progress(f'Exported {name} PNG/TIFF')
        # research-ready composite display, downsampled only for layout, originals remain untouched
        items=[n for n in ['IPF_X','IPF_Y','IPF_Z','Phase','KAM','BC','MAD','PQ'] if (singles/(n+'.png')).exists()]
        rows,cols=style.get('layout',(2,4))
        if rows<1 or cols<1 or rows*cols<len(items):
            raise ValueError('Layout must have at least as many cells as selected maps')
        fig,axes=plt.subplots(rows,cols,figsize=(4.4*cols,4.6*rows),dpi=dpi,layout='constrained',squeeze=False)
        for ax,name in zip(axes.ravel(),items):
            im=Image.open(singles/(name+'.png'))
            # Publication raster is never downsampled before vector embedding.
            # Native pixels are retained, with their measured physical aspect.
            ax.imshow(im,interpolation='none',aspect=sy/sx)
            ax.set_title('MAD (°)' if name=='MAD' else name.replace('_','-'),fontsize=14,fontweight='bold')
            if name=='MAD':
                fig.colorbar(ScalarMappable(norm=Normalize(0,2),cmap='magma'),ax=ax,
                             shrink=0.75,ticks=[0,1,2],label='MAD (°)')
            ax.axis('off')
            length_um=50 if w*sx>=100 else max(1,int(w*sx//4))
            length_px=im.width*length_um/(w*sx)
            # Image fill is entire axis, match thumbnail pixels
            ax.plot([im.width*0.68,im.width*0.68+length_px],
                    [im.height*0.94]*2,color='white',lw=3,solid_capstyle='butt')
            ax.text(im.width*0.68+length_px/2,im.height*0.91,f'{length_um:g} µm',
                    color='white',fontsize=9,ha='center',weight='bold')
        for ax in list(axes.ravel())[len(items):]:ax.axis('off')
        fig.suptitle(f'EBSD montage | {w}×{h}, step {sx:g} × {sy:g} µm | IPF: orix TSL convention, NOT AZtec calibrated',fontsize=12)
        for fmt in formats:
            check_cancelled()
            progress(f'Exporting {fmt.upper()} paper layout')
            fig.savefig(combined/('EBSD_paper_layout.'+fmt),dpi=dpi)
        plt.close(fig)
        if mad_unit is not None:
            legend_fig,legend_ax=plt.subplots(figsize=(4.2,1.2),dpi=180,layout='constrained')
            legend_fig.colorbar(ScalarMappable(norm=Normalize(0,2),cmap='magma'),cax=legend_ax,
                                orientation='horizontal',ticks=[0,0.5,1,1.5,2],label='MAD (°)')
            for fmt in formats:
                progress(f'Exporting {fmt.upper()} MAD colourbar')
                legend_fig.savefig(legends/('MAD_colourbar.'+fmt),dpi=dpi)
            plt.close(legend_fig)
        # colour-scale descriptions, explicit colour-key provenance
        doc={'Phase':phase_colors,'KAM':{'low_deg':0,'high_deg':5,'map':'jet','algorithm':'4-neighbour, symmetry-reduced'},
             'BC':{'low':0,'high':255,'map':'gray'},'MAD':{'low_deg':0,'high_deg':2,'map':'magma','input_unit':mad_unit,
                    'display_unit':'deg','conversion':'rad * 180/pi' if mad_unit=='rad' else 'identity',
                    'invalid':'black for unindexed, nonfinite or negative MAD',
                    'scale_policy':'fixed project display range; not calibrated to AZtec'},
             'IPF':'orix IPFColorKeyTSL; cannot claim identical to Oxford AZtec without external calibration'}
        import json
        (outdir/'colour_legend_metadata.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
        (outdir/'render_timings.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
        progress('Rendering complete')
        return combined/('EBSD_paper_layout.png' if 'png' in formats else 'EBSD_paper_layout.'+formats[0])
