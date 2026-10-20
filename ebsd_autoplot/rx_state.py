"""Configurable GOS/KAM/GROD recrystallization *proxy*, not ground-truth RX."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

STATE_NAMES = {0: "unclassified", 1: "recrystallized_proxy", 2: "substructured_proxy", 3: "deformed_proxy"}


@dataclass(frozen=True)
class RxThresholds:
    rx_gos_deg: float = 1.0
    substructured_gos_deg: float = 2.0
    rx_kam_deg: float = 0.5
    substructured_kam_deg: float = 1.0
    rx_grod_p95_deg: float | None = None
    substructured_grod_p95_deg: float | None = None

    def validate(self):
        for low, high in ((self.rx_gos_deg, self.substructured_gos_deg),
                          (self.rx_kam_deg, self.substructured_kam_deg)):
            if not np.isfinite([low, high]).all() or low < 0 or high < low:
                raise ValueError("RX proxy thresholds must be finite, nonnegative and ordered")
        optional = (self.rx_grod_p95_deg, self.substructured_grod_p95_deg)
        if any(v is not None for v in optional):
            if any(v is None for v in optional) or not np.isfinite(optional).all() or min(optional) < 0 or optional[1] < optional[0]:
                raise ValueError("Both GROD thresholds must be finite, nonnegative and ordered")
        return self


def classify_rx(grains, labels, phase, indexed, phase_specs, *, thresholds=None,
                min_phase_grains=1, min_beta_grains=5, min_beta_pixels=50,
                min_indexed_fraction=0.0):
    """Classify existing grains; provide count/area/all-ROI/phase denominators.

    Threshold mappings use original phase IDs. No phase ID is assumed to be Ti.
    Unclassified grains are excluded from the classified denominators and are
    explicitly retained in all-ROI and indexed-phase area denominators.
    """
    labels, phase, indexed = np.asarray(labels), np.asarray(phase), np.asarray(indexed, bool)
    if labels.shape != phase.shape or labels.shape != indexed.shape:
        raise ValueError("RX input maps must share one ROI")
    if not 0 <= float(min_indexed_fraction) <= 1:
        raise ValueError("Minimum indexing fraction must lie in [0,1]")
    if min(int(min_phase_grains), int(min_beta_grains), int(min_beta_pixels)) < 1:
        raise ValueError("Phase sample thresholds must be positive")
    thresholds = {} if thresholds is None else thresholds
    states = np.zeros(labels.shape, np.uint8)
    phase_rows = []
    assignments = {}
    all_indexed_fraction = float(indexed.mean())
    for raw_pid, spec in phase_specs.items():
        pid = int(raw_pid)
        cfg = thresholds.get(pid, thresholds.get(str(pid), RxThresholds()))
        if isinstance(cfg, dict): cfg = RxThresholds(**cfg)
        cfg.validate()
        group = [g for g in grains if g["phase_id"] == pid]
        eligible = [g for g in group if g["qc_included"] and np.isfinite(g["GOS_deg"]) and
                    np.isfinite(g["KAM_mean_deg"]) and g["KAM_valid_count"] > 0]
        beta = int(spec.get("space_group", 0)) == 229 and any(
            name in str(spec.get("name", "")).lower() for name in ("beta", "β", "bcc", "titanium-cub"))
        phase_pixels = int(np.count_nonzero(indexed & (phase == pid)))
        reason = "available_proxy_requires_material_calibration"
        if all_indexed_fraction < min_indexed_fraction: reason = "unavailable_low_indexing"
        elif len(eligible) < (min_beta_grains if beta else min_phase_grains): reason = "unavailable_insufficient_grains"
        elif beta and phase_pixels < min_beta_pixels: reason = "unavailable_insufficient_beta_pixels"
        if reason.startswith("available"):
            for grain in eligible:
                gos, kam = grain["GOS_deg"], grain["KAM_mean_deg"]
                p95 = grain["GROD_p95_deg"]
                if cfg.rx_grod_p95_deg is not None and not np.isfinite(p95):
                    continue
                low = gos <= cfg.rx_gos_deg and kam <= cfg.rx_kam_deg
                mid = gos <= cfg.substructured_gos_deg and kam <= cfg.substructured_kam_deg
                if cfg.rx_grod_p95_deg is not None:
                    low &= p95 <= cfg.rx_grod_p95_deg
                    mid &= p95 <= cfg.substructured_grod_p95_deg
                assignments[grain["grain_id"]] = 1 if low else 2 if mid else 3
        classified = [g for g in group if assignments.get(g["grain_id"], 0)]
        classified_pixels = sum(g["pixels"] for g in classified)
        for state, name in STATE_NAMES.items():
            subset = [g for g in group if assignments.get(g["grain_id"], 0) == state]
            pixels = sum(g["pixels"] for g in subset) if state else phase_pixels - classified_pixels
            available = reason.startswith("available") and state > 0
            phase_rows.append({"phase_id": pid, "phase_name": spec.get("name", str(pid)),
                "state_code": state, "state": name, "status": reason,
                "grain_count": len(subset), "pixel_count": pixels,
                "classified_grain_denominator": len(classified), "classified_pixel_denominator": classified_pixels,
                "all_ROI_pixel_denominator": phase.size, "indexed_phase_pixel_denominator": phase_pixels,
                "number_fraction_classified": len(subset) / len(classified) if available and classified else float("nan"),
                "area_fraction_classified": pixels / classified_pixels if available and classified_pixels else float("nan"),
                "area_fraction_all_ROI": pixels / phase.size if available or state == 0 else float("nan"),
                "area_fraction_indexed_phase": pixels / phase_pixels if (available or state == 0) and phase_pixels else float("nan"),
                "thresholds_json": __import__("json").dumps(asdict(cfg), sort_keys=True)})
    # Sparse label values are legal: allocate by the number of unique labels,
    # never by max(label), and never scan the entire image once per grain.
    unique, inverse = np.unique(labels, return_inverse=True)
    lookup = np.array([assignments.get(int(gid), 0) for gid in unique], np.uint8)
    states = lookup[inverse].reshape(labels.shape)
    states[~indexed] = 0
    return states, phase_rows, assignments


def threshold_sensitivity(grains, labels, phase, indexed, phase_specs, *, thresholds=None,
                          scales=(0.8, 1.0, 1.2), **kwargs):
    rows = []
    for scale in scales:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Sensitivity multipliers must be positive")
        scaled = {}
        for pid in phase_specs:
            value = (thresholds or {}).get(int(pid), (thresholds or {}).get(str(pid), RxThresholds()))
            data = asdict(RxThresholds(**value)) if isinstance(value, dict) else asdict(value)
            scaled[int(pid)] = {k: (None if v is None else v * scale) for k, v in data.items()}
        _, summary, _ = classify_rx(grains, labels, phase, indexed, phase_specs, thresholds=scaled, **kwargs)
        rows.extend(dict(row, threshold_scale=float(scale)) for row in summary)
    return rows
