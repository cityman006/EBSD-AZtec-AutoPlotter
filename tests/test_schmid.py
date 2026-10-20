import numpy as np

from ebsd_autoplot.schmid import _load_axis, slip_systems_alpha


def test_alpha_slip_families_have_orthogonal_systems_and_expected_counts():
    systems = slip_systems_alpha(4.68 / 2.95)
    assert {k: len(v) for k, v in systems.items()} == {
        "basal_a": 3, "prismatic_a": 3, "pyramidal_a": 6, "pyramidal_c+a": 12
    }
    for family in systems.values():
        for item in family:
            assert abs(item["s_dot_n"]) < 1e-8
            assert np.isclose(np.linalg.norm(item["direction"]), 1)
            assert np.isclose(np.linalg.norm(item["normal"]), 1)


def test_loading_axis_is_normalized_and_invalid_axes_rejected():
    assert np.allclose(_load_axis([0, 0, 3]), [0, 0, 1])
    for axis in ([0, 0, 0], [float("nan"), 0, 1], [1, 2]):
        try:
            _load_axis(axis)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid load axis accepted: {axis}")
