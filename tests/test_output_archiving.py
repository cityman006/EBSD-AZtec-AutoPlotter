import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py

from ebsd_autoplot.archive import create_context, preview_legacy, import_legacy, write_run_manifest
from ebsd_autoplot.cli import main
from ebsd_autoplot.compute import prepare
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.render import maps
from test_pipeline import write_synth


def _make_source(folder, name='sample.h5oina'):
    write_synth(folder)
    return H5Source(folder / name)


def test_same_source_gets_new_run_and_different_source_isolated(tmp_path):
    a = _make_source(tmp_path / 'A')
    b = _make_source(tmp_path / 'B')
    root = tmp_path / 'results'
    first = create_context(a, root)
    second = create_context(a, root)
    other = create_context(b, root)
    assert first.dataset_dir == second.dataset_dir
    assert first.analysis_run_id != second.analysis_run_id
    assert first.analysis_dir != second.analysis_dir
    assert first.dataset_dir != other.dataset_dir
    assert first.dataset_id != other.dataset_id
    assert first.cache_dir.parent == first.analysis_dir
    assert first.render_dir.parent.name == 'renders'


def test_concurrent_contexts_reserve_unique_run_dirs(tmp_path):
    src = _make_source(tmp_path / 'A')
    root = tmp_path / 'results'
    with ThreadPoolExecutor(max_workers=4) as pool:
        contexts = list(pool.map(lambda _: create_context(src, root), range(4)))
    assert len({x.analysis_run_id for x in contexts}) == 4
    assert len({x.render_id for x in contexts}) == 4


def test_source_change_keeps_dataset_identity_but_changes_version(tmp_path):
    src_path = tmp_path / 'A' / 'sample.h5oina'
    src = _make_source(src_path.parent)
    first = create_context(src, tmp_path / 'results')
    with h5py.File(src_path, 'a') as f: f.attrs['revision'] = 'changed'
    second = create_context(H5Source(src_path), tmp_path / 'results')
    assert first.dataset_id == second.dataset_id
    assert first.source_version != second.source_version


def test_render_only_uses_selected_cache_without_source_read_and_keeps_cache(tmp_path, monkeypatch):
    src = _make_source(tmp_path / 'A')
    context = create_context(src, tmp_path / 'results')
    cache = prepare(src, context.cache_dir, batch=4, do_ipf=False, progress=lambda _: None)
    write_run_manifest(context.analysis_dir/'analysis_manifest.json', {
        'status':'success','dataset_id':context.dataset_id,'analysis_run_id':context.analysis_run_id,
        'scan_id':src.scan_path,'source_version':context.source_version})
    before = hashlib.sha256(cache.read_bytes()).hexdigest()

    def forbidden(*args, **kwargs): raise AssertionError('Render Only must not read or analyze the source')
    monkeypatch.setattr('ebsd_autoplot.cli.H5Source', forbidden)
    monkeypatch.setattr('ebsd_autoplot.cli.prepare', forbidden)
    monkeypatch.setattr('ebsd_autoplot.cli.run_v2', forbidden)
    assert main(['--cache', str(cache), '--font', 'Arial', '--formats', 'png', 'pdf', 'svg']) == 0
    after = hashlib.sha256(cache.read_bytes()).hexdigest()
    assert before == after
    renders = list(context.dataset_dir.joinpath('renders').glob('render_*'))
    assert len(renders) == 2  # reserved render directory + completed render-only directory
    completed = next(p for p in renders if (p / 'render_manifest.json').exists())
    manifest = json.loads((completed / 'render_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'success'
    assert manifest['analysis_run_id'] == context.analysis_run_id
    assert (completed / 'combined' / 'EBSD_paper_layout.pdf').is_file()
    assert (completed / 'combined' / 'EBSD_paper_layout.svg').is_file()
    svg = (completed / 'combined' / 'EBSD_paper_layout.svg').read_text(encoding='utf-8')
    assert '<text' in svg  # editable text mode


def test_missing_font_fails_instead_of_silent_fallback(tmp_path):
    src = _make_source(tmp_path / 'A')
    context = create_context(src, tmp_path / 'results')
    cache = prepare(src, context.cache_dir, batch=4, do_ipf=False, progress=lambda _: None)
    try:
        maps(cache, tmp_path / 'render', style={'font': 'Definitely Missing EBSD Font'})
    except ValueError as exc:
        assert 'not installed' in str(exc)
    else:
        raise AssertionError('Missing font silently fell back')


def test_legacy_preview_and_copy_preserve_old_files(tmp_path):
    old=tmp_path/'old_output';known=old/'run_a';known.mkdir(parents=True)
    source=tmp_path/'source'/'legacy.h5oina';source.parent.mkdir()
    (known/'source_metadata.json').write_text(json.dumps({'source':str(source),'scan_path':'/1/EBSD'}),encoding='utf-8')
    (known/'IPF_X.png').write_bytes(b'old image')
    (old/'IPF_X.png').write_bytes(b'unknown image')
    (old/'raw.h5oina').write_bytes(b'raw input marker')
    preview=preview_legacy(old)
    assert preview['confirmed'][0]['status']=='confirmed'
    assert len(preview['unassigned_files'])==1
    assert len(preview['ignored_measurement_files'])==1
    output=tmp_path/'new_results'
    copied=import_legacy(old,output,include_unassigned=True)
    assert len(copied)==2
    assert (known/'IPF_X.png').read_bytes()==b'old image'
    assert (old/'raw.h5oina').read_bytes()==b'raw input marker'
    known_import=next(p for p in copied if p.name.startswith('render_'))
    assert (known_import/'IPF_X.png').read_bytes()==b'old image'
    unknown_import=next(p for p in copied if p.name.startswith('collection_'))
    assert list(unknown_import.rglob('IPF_X.png'))[0].read_bytes()==b'unknown image'
