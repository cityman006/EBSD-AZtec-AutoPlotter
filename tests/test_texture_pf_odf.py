import numpy as np
from scipy.spatial.transform import Rotation

from ebsd_autoplot.texture_v3 import _odf_grid, _pole_family


def test_pf_families_and_mrd_normalization():
    for sg, families in ((194, ("0001", "10-10", "11-20")),
                         (229, ("100", "110", "111"))):
        for family in families:
            poles = _pole_family(sg, family, 4.68 / 2.95)
            assert poles.ndim == 2 and poles.shape[1] == 3
            assert np.allclose(np.linalg.norm(poles, axis=1), 1)
    samples = Rotation.from_euler("ZXZ", np.zeros((50, 3))).as_quat()
    phi1, Phi, phi2, mrd, params = _odf_grid(samples, np.ones(50), 194, 30, 90, 7)
    assert mrd.shape == (len(phi1), len(Phi), len(phi2))
    assert np.all(np.isfinite(mrd)) and np.all(mrd >= 0)
    assert abs(params["integral_check"] - 1) < 1e-5


def test_odf_rejects_too_few_orientations():
    samples = Rotation.from_euler("ZXZ", np.zeros((49, 3))).as_quat()
    try:
        _odf_grid(samples, np.ones(49), 194, 10, 15, 0)
    except ValueError as exc:
        assert "at least 50" in str(exc)
    else:
        raise AssertionError("ODF must reject insufficient sampling")
