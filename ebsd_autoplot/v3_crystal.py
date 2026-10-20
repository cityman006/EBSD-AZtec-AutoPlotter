"""Validated low-level crystallographic conventions used by the V3 research tools.

Orientations are SciPy xyzw unit quaternions mapping crystal Cartesian vectors
to sample Cartesian vectors. Euler input follows the existing Bunge ZXZ/radian
contract. No inference is made from a phase number alone.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

ORIENTATION_CONVENTION = "Bunge ZXZ radians; active crystal-Cartesian to sample-Cartesian; quaternion xyzw"


def classify_phase_specs(phase_specs: dict) -> dict[int, str]:
    """Return {raster phase id: alpha|beta}; reject ambiguous phase metadata."""
    result = {}
    for raw_id, spec in phase_specs.items():
        pid = int(raw_id)
        sg = int(spec.get("space_group", 0))
        name = str(spec.get("name", "")).casefold()
        alpha = sg == 194 and any(term in name for term in ("alpha", "α", "hcp", "titanium-hex"))
        beta = sg == 229 and any(term in name for term in ("beta", "β", "bcc", "titanium-cub"))
        if alpha == beta:
            raise ValueError(
                f"Phase {pid} cannot be classified safely: name={spec.get('name')!r}, "
                f"space_group={sg}; expected named HCP Ti (194) or BCC Ti (229)."
            )
        result[pid] = "alpha" if alpha else "beta"
    return result


def normalize_quaternions(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4 or not np.all(np.isfinite(q)):
        raise ValueError("Orientations must be finite xyzw quaternions")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Zero-length quaternion is not an orientation")
    return q / norm


def point_group(space_group: int) -> Rotation:
    if int(space_group) == 194:
        return Rotation.create_group("D6")
    if int(space_group) == 229:
        return Rotation.create_group("O")
    raise ValueError(f"V3 symmetry is not validated for space group {space_group}")


def disorientation_deg(a, b, space_group: int) -> float:
    """Minimum angle under the same crystal point group on both sides."""
    qa = normalize_quaternions(np.asarray(a).reshape(1, 4))[0]
    qb = normalize_quaternions(np.asarray(b).reshape(1, 4))[0]
    sym = point_group(space_group)
    rel = Rotation.from_quat(qa).inv() * Rotation.from_quat(qb)
    # The minimum rotation angle is unchanged by quaternion sign. Enumerating
    # the full double action is explicit and avoids silently using a one-sided
    # symmetry reduction in a scientific fit.
    qrel = rel.as_quat()
    qsym = sym.as_quat()
    # Batched quaternion product q_left * q_rel * q_right.
    lv, lw = qsym[:, None, :3], qsym[:, None, 3:4]
    rv, rw = qsym[None, :, :3], qsym[None, :, 3:4]
    v, w = qrel[:3], qrel[3]
    mid_v = lw * v + w * lv + np.cross(lv, np.broadcast_to(v, lv.shape))
    mid_w = lw * w - np.sum(lv * v, axis=-1, keepdims=True)
    out_v = mid_w * rv + rw * mid_v + np.cross(mid_v, rv)
    out_w = mid_w * rw - np.sum(mid_v * rv, axis=-1, keepdims=True)
    scalar = np.clip(np.abs(out_w[..., 0]), 0.0, 1.0)
    return float(np.degrees(2 * np.arccos(np.max(scalar))))


def hcp_direction_cartesian(direction_4, c_over_a: float) -> np.ndarray:
    """Convert a Miller-Bravais direction [u v t w] into Cartesian coordinates."""
    u, v, t, w = map(float, direction_4)
    if not np.isclose(u + v + t, 0):
        raise ValueError("Miller-Bravais directions require u+v+t=0")
    if not np.isfinite(c_over_a) or c_over_a <= 0:
        raise ValueError("A positive measured c/a ratio is required for HCP directions")
    # a1=(1,0), a2=(-1/2,sqrt(3)/2); W=w/3 is the standard
    # four-index to three-index direction conversion.
    U, V, W = (2 * u + v) / 3, (u + 2 * v) / 3, w / 3
    return np.array([U - V / 2, np.sqrt(3) * V / 2, c_over_a * W], dtype=np.float64)


def hcp_plane_normal_cartesian(plane_4, c_over_a: float) -> np.ndarray:
    """Reciprocal-metric normal for (h k i l), with a1/a2 120 degrees apart."""
    h, k, i, l = map(float, plane_4)
    if not np.isclose(h + k + i, 0):
        raise ValueError("Miller-Bravais planes require h+k+i=0")
    if not np.isfinite(c_over_a) or c_over_a <= 0:
        raise ValueError("A positive measured c/a ratio is required for HCP planes")
    # Reciprocal vectors b1=(1,1/sqrt(3)), b2=(0,2/sqrt(3)).
    n = np.array([h, (h + 2 * k) / np.sqrt(3), l / c_over_a], dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Zero Miller-Bravais plane normal")
    return n / norm


def canonical_axis(v, *, tolerance=1e-7) -> np.ndarray:
    """Normalize a projective axis (v and -v represent the same plane/direction)."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot canonicalize a zero vector")
    v = v / n
    for x in v:
        if abs(x) > tolerance:
            if x < 0:
                v = -v
            break
    return v
