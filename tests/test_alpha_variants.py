import numpy as np

from ebsd_autoplot.alpha_variants import (burgers_geometry_check, burgers_variant_orientations,
                                          candidate_parent_orientations, assign_alpha_grains,
                                          _variant_boundary_types)


def test_burgers_relation_has_twelve_variants_and_geometry():
    assert len(burgers_variant_orientations()) == 12
    check = burgers_geometry_check()
    assert check["variant_count"] == 12
    assert check["beta_plane_alpha_c_misalignment_deg"] < 1e-5
    assert check["beta_direction_alpha_a_misalignment_deg"] < 1e-5


def test_isolated_alpha_grain_remains_unknown_and_measured_beta_anchor_is_explicit():
    q = [0, 0, 0, 1]
    measured_parent = candidate_parent_orientations(q, burgers_variant_orientations())[0].as_quat().tolist()
    grains = [{"grain_id": 1, "phase_id": 1, "q": q},
              {"grain_id": 2, "phase_id": 2, "q": measured_parent}]
    isolated = assign_alpha_grains(grains[:1], [], alpha_phase_id=1, beta_phase_id=2)
    assert isolated[1]["status"] == "unknown"
    anchored = assign_alpha_grains(grains, [(1, 2, 10)], beta_anchors={2: measured_parent},
                                   alpha_phase_id=1, beta_phase_id=2)
    assert anchored[1]["beta_anchor_used"]


def test_variant_boundary_map_distinguishes_parent_and_unknown_edges():
    grains = np.array([[1, 2, 3, 4]])
    parents = np.array([[10, 10, 20, 0]])
    result = _variant_boundary_types(grains, parents)
    assert result.tolist() == [[1, 2, 3, 3]]
