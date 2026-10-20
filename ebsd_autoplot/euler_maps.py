"""Bunge ZXZ Euler scalar maps and a separately labelled Euler RGB display."""
from __future__ import annotations

import warnings
import numpy as np
from scipy.spatial.transform import Rotation

from .v3_crystal import ORIENTATION_CONVENTION, normalize_quaternions

EULER_DISPLAY_CONVENTION = (ORIENTATION_CONVENTION +
    "; phi1/phi2 in [0,360), Phi in [0,180]; RGB=(phi1/360,Phi/180,phi2/360); RGB is not IPF")


def quaternion_euler_maps(quaternions, indexed):
    """Return degrees and RGB, NaN/black at nonindexed or invalid orientations.

    Gimbal lock has nonunique Euler triplets. SciPy sets phi2=0 while retaining
    the same physical rotation. No symmetry reduction is applied to Euler RGB.
    """
    q, indexed = np.asarray(quaternions, float), np.asarray(indexed, bool)
    if q.shape != (*indexed.shape, 4):
        raise ValueError("Expected xyzw quaternion map of shape (H,W,4)")
    good = indexed & np.isfinite(q).all(axis=-1) & (np.linalg.norm(q, axis=-1) > 1e-12)
    deg = np.full((*indexed.shape, 3), np.nan, np.float32)
    if good.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
            angles = Rotation.from_quat(normalize_quaternions(q[good])).as_euler("ZXZ", degrees=True)
        angles[:, [0, 2]] %= 360
        deg[good] = angles
    rgb = np.zeros_like(deg)
    rgb[good] = deg[good] / np.array([360, 180, 360])
    return deg, rgb, good
