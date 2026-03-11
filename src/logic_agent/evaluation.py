"""
evaluation.py  –  MVTec LOCO evaluation metrics
================================================

Implements the two official MVTec LOCO AD metrics:

  1. Image-level AUROC
        Standard ROC-AUC over (gt_label, anomaly_score) pairs.
        One score per image, threshold-free.

  2. Saturated Per-Region Overlap (sPRO)
        The official localisation metric from the MVTec LOCO paper.
        For each defect region in the GT mask the overlap with the
        predicted anomaly map is computed and *saturated* at a
        per-defect threshold defined in defects_config.json:

            overlap_i = min(pixel_overlap_i, sat_threshold_i)
                        / sat_threshold_i

        The sPRO curve is the mean of these capped overlaps swept over
        FPR values [0, fpr_limit].  The scalar metric is the normalised
        area under that curve (divided by fpr_limit).

Ground-truth mask format (MVTec LOCO)
--------------------------------------
Each anomalous test image has a paired GT mask file (uint8 PNG).
Pixel values encode defect identity:
    0          → background (normal)
    pixel_value → belongs to defect class with that label in defects_config.json

defects_config.json fields used here:
    pixel_value        : uint8 label in the GT mask
    saturation_threshold: numeric saturation cap
    relative_saturation : if True, sat_threshold = saturation_threshold
                          × total area of that defect region in the GT mask
                          (always 1.0 in the provided config, so cap = region area,
                           meaning full overlap is required to saturate)
                          if False, sat_threshold is an absolute pixel count.

Image-level label derivation
-----------------------------
An image is anomalous (gt_label = 1) if any non-zero pixel exists in its
GT mask.  Good samples have no GT mask file (or an all-zero mask).

Usage
-----
    from evaluation import LocoEvaluator

    evaluator = LocoEvaluator.from_config("path/to/defects_config.json")

    # During inference loop:
    for sample in test_dataset:
        result = detector.predict(sample["image_np"])
        evaluator.add(
            anomaly_score  = result["total_energy"],
            anomaly_map    = result.get("anomaly_map"),   # [H, W] float32 or None
            gt_mask        = sample["gt_mask"],            # [H, W] uint8  or None
        )

    metrics = evaluator.compute()
    print(metrics["image_auroc"], metrics["spro"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# sklearn is available in the project environment
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Defect config
# ---------------------------------------------------------------------------


@dataclass
class DefectConfig:
    defect_name: str
    pixel_value: int
    saturation_threshold: float
    relative_saturation: bool


def load_defects_config(config_path: str | Path) -> List[DefectConfig]:
    """Parse a MVTec LOCO defects_config.json into a list of DefectConfig."""
    raw = json.loads(Path(config_path).read_text())
    return [
        DefectConfig(
            defect_name=entry["defect_name"],
            pixel_value=int(entry["pixel_value"]),
            saturation_threshold=float(entry["saturation_threshold"]),
            relative_saturation=bool(entry["relative_saturation"]),
        )
        for entry in raw
    ]


# ---------------------------------------------------------------------------
# Per-sample record
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    anomaly_score: float  # scalar detector output (higher = more anomalous)
    gt_label: int  # 0 = good, 1 = anomalous
    anomaly_map: Optional[np.ndarray]  # [H, W] float32 predicted heatmap
    gt_mask: Optional[np.ndarray]  # [H, W] uint8   ground-truth mask
    defect_type: str  # human-readable defect name or "good"
    img_path: str


# ---------------------------------------------------------------------------
# sPRO helpers
# ---------------------------------------------------------------------------


def _saturation_cap(
    region_mask: np.ndarray,
    cfg: DefectConfig,
) -> float:
    """
    Compute the saturation cap for one connected defect region.

    If relative_saturation is True:
        cap = saturation_threshold × region_area
        (with saturation_threshold=1.0 this equals the full region area,
         meaning perfect overlap is required to score 1.0)
    If relative_saturation is False:
        cap = saturation_threshold  (absolute pixel count)
    """
    region_area = float(region_mask.sum())
    if cfg.relative_saturation:
        return cfg.saturation_threshold * region_area
    return cfg.saturation_threshold


def _compute_spro_curve(
    records: List[SampleRecord],
    defect_cfgs: Dict[int, DefectConfig],  # pixel_value → DefectConfig
    n_thresholds: int = 400,
    fpr_limit: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the sPRO curve and its normalised AUC.

    The curve is constructed by sweeping a threshold t over the predicted
    anomaly map values.  For each t:

        binary_pred[i,j] = anomaly_map[i,j] >= t

    Then for each anomalous image and each distinct defect region:

        overlap = (binary_pred & gt_region).sum()
        sat_overlap = min(overlap, sat_cap) / sat_cap

    sPRO(t) = mean over all regions across all images.

    FPR(t) is computed over all *good* pixels across good images.

    Returns
    -------
    fprs     : np.ndarray [n_thresholds]
    spros    : np.ndarray [n_thresholds]
    auc_norm : float  –  AUC normalised by fpr_limit  (official metric)
    """
    # Collect all anomaly map values to define the threshold sweep
    all_scores = []
    for rec in records:
        if rec.anomaly_map is not None:
            all_scores.append(rec.anomaly_map.ravel())
    if not all_scores:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    all_vals = np.concatenate(all_scores)
    thresholds = np.linspace(all_vals.min(), all_vals.max(), n_thresholds + 1)

    # Pre-extract GT regions per anomalous sample
    # Each entry: list of (binary_region_mask, sat_cap)
    anomalous_regions: List[List[Tuple[np.ndarray, float]]] = []
    for rec in records:
        if rec.gt_label != 1 or rec.gt_mask is None or rec.anomaly_map is None:
            continue
        regions_for_image: List[Tuple[np.ndarray, float]] = []
        for pv, cfg in defect_cfgs.items():
            class_mask = rec.gt_mask == pv
            if not class_mask.any():
                continue
            # Each connected component is treated as one region
            import cv2

            n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
                class_mask.astype(np.uint8), connectivity=8
            )
            for lbl in range(1, n_labels):  # skip background (0)
                region = label_map == lbl
                cap = _saturation_cap(region, cfg)
                if cap > 0:
                    regions_for_image.append((region, cap))
        if regions_for_image:
            anomalous_regions.append(regions_for_image)

    # Pre-build good-pixel masks (for FPR computation)
    good_pixel_masks: List[np.ndarray] = []
    for rec in records:
        if rec.gt_label == 0 and rec.anomaly_map is not None:
            # Good pixels = everywhere (no gt mask, so all pixels are normal)
            good_pixel_masks.append(rec.anomaly_map)

    # Pair each anomalous image's anomaly_map with its regions
    anomalous_maps_and_regions: List[Tuple[np.ndarray, List[Tuple[np.ndarray, float]]]] = []
    anom_idx = 0
    for rec in records:
        if rec.gt_label != 1 or rec.gt_mask is None or rec.anomaly_map is None:
            continue
        if anom_idx < len(anomalous_regions) and anomalous_regions[anom_idx]:
            anomalous_maps_and_regions.append((rec.anomaly_map, anomalous_regions[anom_idx]))
        anom_idx += 1

    fprs: List[float] = []
    spros: List[float] = []

    total_good_pixels = sum(m.size for m in good_pixel_masks)

    for t in thresholds:
        # ── FPR ────────────────────────────────────────────────────────
        if total_good_pixels > 0:
            fp_pixels = sum(int((m >= t).sum()) for m in good_pixel_masks)
            fpr = fp_pixels / total_good_pixels
        else:
            fpr = 0.0

        # ── sPRO ───────────────────────────────────────────────────────
        sat_overlaps: List[float] = []
        for amap, regions in anomalous_maps_and_regions:
            pred_bin = amap >= t
            for region, cap in regions:
                overlap = float((pred_bin & region).sum())
                sat_overlap = min(overlap, cap) / cap
                sat_overlaps.append(sat_overlap)

        spro = float(np.mean(sat_overlaps)) if sat_overlaps else 0.0

        fprs.append(fpr)
        spros.append(spro)

    fprs = np.array(fprs, dtype=np.float64)
    spros = np.array(spros, dtype=np.float64)

    # Sort by FPR (thresholds go high→low, so FPR goes low→high)
    order = np.argsort(fprs)
    fprs = fprs[order]
    spros = spros[order]

    # Normalised AUC under the curve up to fpr_limit
    mask = fprs <= fpr_limit
    if mask.sum() < 2:
        auc_norm = 0.0
    else:
        auc_norm = float(np.trapz(spros[mask], fprs[mask]) / fpr_limit)

    return fprs, spros, auc_norm


# ---------------------------------------------------------------------------
# Per-defect-type image-level accuracy
# ---------------------------------------------------------------------------


def _per_defect_stats(
    records: List[SampleRecord],
    threshold: float,
) -> Dict[str, Dict]:
    """
    For a fixed decision threshold on anomaly_score, compute per-defect-type
    accuracy, precision, recall, and counts.
    """
    stats: Dict[str, Dict] = {}

    for rec in records:
        dt = rec.defect_type
        if dt not in stats:
            stats[dt] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0}

        pred = int(rec.anomaly_score >= threshold)
        gt = rec.gt_label

        stats[dt]["n"] += 1
        if gt == 1 and pred == 1:
            stats[dt]["tp"] += 1
        elif gt == 0 and pred == 1:
            stats[dt]["fp"] += 1
        elif gt == 0 and pred == 0:
            stats[dt]["tn"] += 1
        else:
            stats[dt]["fn"] += 1

    def _div(a, b):
        return a / b if b > 0 else 0.0

    result = {}
    for dt, s in sorted(stats.items()):
        tp, fp, tn, fn = s["tp"], s["fp"], s["tn"], s["fn"]
        prec = _div(tp, tp + fp)
        rec_ = _div(tp, tp + fn)
        result[dt] = {
            "n": s["n"],
            "n_anomaly": tp + fn,
            "n_good": tn + fp,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": _div(tp + tn, s["n"]),
            "precision": prec,
            "recall": rec_,
            "f1": _div(2 * prec * rec_, prec + rec_),
        }

    return result


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class LocoEvaluator:
    """
    Accumulates per-image predictions and computes MVTec LOCO metrics.

    Metrics
    -------
    image_auroc : float  [0, 1]
        Standard ROC-AUC on (gt_label, anomaly_score).

    spro        : float  [0, 1]
        Normalised AUC of the sPRO curve up to FPR=fpr_limit.
        Uses per-defect saturation caps from defects_config.json.
        Requires anomaly_map and gt_mask to be provided for each sample.

    accuracy    : float  [0, 1]
        Image-level accuracy at the optimal F1 threshold (Youden's J).

    per_defect  : Dict[str, Dict]
        Per-defect-type breakdown (accuracy, precision, recall, F1, counts).

    Parameters
    ----------
    defect_cfgs : list of DefectConfig (from load_defects_config)
    fpr_limit   : float  upper FPR bound for sPRO AUC normalisation (default 0.3)
    n_thresholds: int    number of threshold steps for sPRO curve
    """

    def __init__(
        self,
        defect_cfgs: List[DefectConfig],
        fpr_limit: float = 0.3,
        n_thresholds: int = 400,
    ):
        self.defect_cfgs = defect_cfgs
        self.fpr_limit = fpr_limit
        self.n_thresholds = n_thresholds
        # pixel_value → DefectConfig  (fast lookup during sPRO computation)
        self._pv_map: Dict[int, DefectConfig] = {cfg.pixel_value: cfg for cfg in defect_cfgs}
        self._records: List[SampleRecord] = []

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        fpr_limit: float = 0.3,
        n_thresholds: int = 400,
    ) -> "LocoEvaluator":
        """Construct from a defects_config.json path."""
        cfgs = load_defects_config(config_path)
        return cls(cfgs, fpr_limit=fpr_limit, n_thresholds=n_thresholds)

    # ------------------------------------------------------------------
    def add(
        self,
        anomaly_score: float,
        gt_mask: Optional[np.ndarray],  # [H, W] uint8  or None for good images
        anomaly_map: Optional[np.ndarray] = None,  # [H, W] float32 or None
        defect_type: str = "unknown",
        img_path: str = "",
    ) -> None:
        """
        Register one test image result.

        Parameters
        ----------
        anomaly_score : scalar detector output (e.g. total_energy)
        gt_mask       : uint8 ground-truth mask from MVTec LOCO.
                        None or all-zero → good image (gt_label = 0).
        anomaly_map   : float32 [H, W] spatial heatmap; required for sPRO.
                        Can be None (sPRO will be skipped for this sample).
        defect_type   : defect category string (for per-type breakdown)
        img_path      : source file path (for debugging)
        """
        # Derive image-level binary label from GT mask
        if gt_mask is None or not np.any(gt_mask > 0):
            gt_label = 0
        else:
            gt_label = 1

        self._records.append(
            SampleRecord(
                anomaly_score=float(anomaly_score),
                gt_label=gt_label,
                anomaly_map=anomaly_map,
                gt_mask=gt_mask,
                defect_type=defect_type,
                img_path=img_path,
            )
        )

    # ------------------------------------------------------------------
    def compute(self) -> Dict:
        """
        Compute all metrics over accumulated records.

        Returns
        -------
        dict with keys:
            image_auroc   : float
            spro          : float   (NaN if no anomaly maps provided)
            spro_curve    : dict with 'fprs' and 'spros' arrays
            accuracy      : float   (at optimal threshold)
            precision     : float
            recall        : float
            f1            : float
            optimal_threshold : float
            tp, fp, tn, fn : int
            n_total       : int
            n_good        : int
            n_anomaly     : int
            per_defect    : Dict[str, Dict]
        """
        if not self._records:
            raise RuntimeError("No records added; call add() before compute().")

        scores = np.array([r.anomaly_score for r in self._records], dtype=np.float64)
        labels = np.array([r.gt_label for r in self._records], dtype=np.int32)

        n_total = len(self._records)
        n_anomaly = int(labels.sum())
        n_good = n_total - n_anomaly

        # ── Image-level AUROC ──────────────────────────────────────────
        if n_anomaly == 0 or n_good == 0:
            image_auroc = float("nan")
        else:
            image_auroc = float(roc_auc_score(labels, scores))

        # ── Optimal threshold via Youden's J  (maximises TPR - FPR) ───
        if n_anomaly > 0 and n_good > 0:
            fprs_roc, tprs_roc, thresh_roc = roc_curve(labels, scores)
            j_scores = tprs_roc - fprs_roc
            best_idx = int(np.argmax(j_scores))
            opt_threshold = float(thresh_roc[best_idx])
        else:
            opt_threshold = float(np.median(scores))

        # ── Image-level confusion matrix at optimal threshold ──────────
        preds = (scores >= opt_threshold).astype(np.int32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        def _div(a, b):
            return float(a) / float(b) if b > 0 else 0.0

        accuracy = _div(tp + tn, n_total)
        precision = _div(tp, tp + fp)
        recall = _div(tp, tp + fn)
        f1 = _div(2 * precision * recall, precision + recall)

        # ── sPRO ──────────────────────────────────────────────────────
        has_maps = any(r.anomaly_map is not None and r.gt_mask is not None for r in self._records)
        if has_maps and n_anomaly > 0:
            fprs_spro, spros_spro, spro_auc = _compute_spro_curve(
                self._records,
                self._pv_map,
                n_thresholds=self.n_thresholds,
                fpr_limit=self.fpr_limit,
            )
        else:
            fprs_spro = np.array([0.0, 1.0])
            spros_spro = np.array([0.0, 0.0])
            spro_auc = float("nan")

        # ── Per-defect breakdown ───────────────────────────────────────
        per_defect = _per_defect_stats(self._records, threshold=opt_threshold)

        return {
            "image_auroc": image_auroc,
            "spro": spro_auc,
            "spro_curve": {
                "fprs": fprs_spro,
                "spros": spros_spro,
            },
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "optimal_threshold": opt_threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n_total": n_total,
            "n_good": n_good,
            "n_anomaly": n_anomaly,
            "per_defect": per_defect,
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all accumulated records."""
        self._records.clear()
