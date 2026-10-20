import csv
import json

from ebsd_autoplot.alpha_variants import analyze_alpha_variants
from ebsd_autoplot.compute import prepare
from ebsd_autoplot.io import H5Source
from ebsd_autoplot.prior_beta import reconstruct_prior_beta
from ebsd_autoplot.v2 import run_v2
from test_pipeline import write_synth


def test_inferred_prior_beta_records_coverage_and_nonmeasurement_label(tmp_path):
    source_dir = tmp_path / "raw"
    write_synth(source_dir, h=10, w=12)
    cache = prepare(H5Source(source_dir / "sample.h5oina"), tmp_path / "cache", batch=4,
                    do_ipf=False, progress=lambda _: None)
    v2 = tmp_path / "cache" / "v2"
    run_v2(cache, v2, roi=[0, 0, 12, 10], min_size=1, progress=lambda _: None)
    variants = tmp_path / "variants"
    analyze_alpha_variants(cache, v2, variants)
    output = tmp_path / "prior_beta"
    qc = reconstruct_prior_beta(cache, v2, variants, output)
    stored = json.loads((output / "reconstruction_qc.json").read_text(encoding="utf-8"))
    assert 0 <= stored["parent_assigned_coverage"] <= 1
    assert stored["parent_volume_fraction"] is None
    assert "inferred" in stored["status"]
    assert (output / "prior_beta_fields.h5").is_file()
    assert (output / "Prior_Beta_IPF.svg").is_file()
    with (output / "parent_grains.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all("not measured" in row["status"] for row in rows)
