import json

from ebsd_autoplot.cli import parser
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.pipeline import analyze
from test_pipeline import write_synth


def test_v3_texture_runs_in_archive_and_records_stage_checkpoint(tmp_path):
    raw = tmp_path / "raw"
    write_synth(raw, h=10, w=12)
    args = parser().parse_args([str(raw / "sample.h5oina"), "--output-root", str(tmp_path / "out"),
                                "--analysis", "v3_texture", "--no-ipf", "--no-render",
                                "--v3-grid-step", "90", "--v3-bandwidth", "30",
                                "--v3-max-samples", "100"])
    run = analyze(H5Source(raw / "sample.h5oina"), tmp_path / "out", vars(args))
    manifest = json.loads((run / "analysis_manifest.json").read_text(encoding="utf-8"))
    cache = run / "analysis_cache" / "features.h5"
    result = run / "analysis_cache" / "v3" / "texture" / "results"
    assert manifest["status"] == "success"
    assert "v3_texture" in manifest["checkpoints"]
    assert (result / "texture_summary.json").is_file()
    assert (result / ".transactions").is_dir()
    assert cache.is_file()
