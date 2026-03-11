"""
pipeline.py  –  HS-CRL end-to-end pipeline
===========================================

Full flow
---------
Input Image
     │
     ▼
SAM2Segmentor          (segment.py)   – hierarchical mask proposals
     │
     ▼
MaskStabilizer         (segment.py)   – area / elongation filtering
     │
     ▼
DINOv3FeatureExtractor (dinov3raw.py) – multi-scale patch features + CLS
     │
     ▼
HierarchicalPartDiscovery (hpd.py)    – intra-region clustering, part graph
     │
     ▼
HSCRL detector         (hs_crl.py)
     │  ├── PrototypeAnomalyMapper →  E_patch
     │  ├── PartTypeAssigner     →  E_object
     │  └── CompositionalRules   →  E_composition
     ▼
E = α·E_patch + β·E_object + γ·E_composition
     │
     ▼
Anomaly decision  (threshold = μ + k·σ of normal scores)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from matplotlib import pyplot as plt

from logic_agent.dataset.mvtec_loco import MVTecLOCODataset
from logic_agent.evaluation import LocoEvaluator
from logic_agent.logging import get_logger
from logic_agent.model.dinov3 import DINO_MODEL, DINOv3FeatureExtractor, visualize_clusters
from logic_agent.model.hpd import HierarchicalPartDiscovery
from logic_agent.model.hs_crl import HSCRL
from logic_agent.model.segment import MODEL_CFG, SAM2_CKPT, MaskStabilizer, SAM2Segmentor, show_anns

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-image processing helpers
# ---------------------------------------------------------------------------
def visualize_segmentation_results(
    image: np.ndarray,
    sam_masks: List[Dict],
    save_path: str | Path,
    show: bool = False,
) -> Path:
    """Visualize and save SAM mask overlays on top of the input image."""
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image)
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].set_title(f"Segmentation Overlay ({len(sam_masks)} masks)")
    axes[1].axis("off")

    cmap = plt.get_cmap("tab20")
    for idx, m in enumerate(sam_masks):
        mask = m.get("segmentation")
        if mask is None:
            continue

        mask_np = mask.astype(bool)
        color = cmap(idx % 20)[:3]
        overlay = np.zeros((*mask_np.shape, 4), dtype=np.float32)
        overlay[mask_np] = [*color, 0.45]
        axes[1].imshow(overlay)

        ys, xs = np.where(mask_np)
        if len(xs) > 0:
            axes[1].text(
                int(xs.mean()),
                int(ys.mean()),
                str(idx),
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.45, boxstyle="round,pad=0.15"),
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    LOGGER.info(f"Saved segmentation visualization: {out_path}")
    return out_path


def _extract_dense_features(
    img_np: np.ndarray,
    dino: DINOv3FeatureExtractor,
) -> torch.Tensor:
    """
    Run DINOv3 on a single image.

    Returns
    -------
    dense_feats : [N, D]   patch-level features (grid_size² patches)
    """
    device = dino.model.device
    inputs = dino.processor(images=img_np, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # [1, C, H, W]

    multi_scale_feats, _, _ = dino.extract_dense_features(pixel_values)

    dense_feats = multi_scale_feats.squeeze(0)  # [N, D]
    return dense_feats


def process_one_image(
    img_np: np.ndarray,
    sam: SAM2Segmentor,
    stabilizer: MaskStabilizer,
    dino: DINOv3FeatureExtractor,
    hpd: HierarchicalPartDiscovery,
    cluster_vis_prefix: Optional[str] = None,
) -> Dict:
    """
    Run the full pipeline on one image.

    Returns a dict consumed by HSCRL.fit() / HSCRL.predict():
        dense_feats  : torch.Tensor [N, D]
        parts        : List[PartDescriptor]
        root         : PartNode  (HPD containment graph root)
        sam_masks    : List[Dict]  (cleaned SAM masks, for visualisation)
    """
    H, W = img_np.shape[:2]

    # 1. SAM2 segmentation
    raw_masks = sam.segment(img_np)
    LOGGER.info(f"SAM2 produced {len(raw_masks)} raw masks")

    # 2. Stability filtering (area, elongation)
    clean_masks = stabilizer.process(raw_masks, H, W)
    LOGGER.info(f"MaskStabilizer retained {len(clean_masks)} masks")

    if cluster_vis_prefix is not None:
        # Reuse visualize_clusters with one cluster per mask to save per-shot overlays.
        lvl1_clusters = {idx: [m] for idx, m in enumerate(clean_masks)}
        visualize_clusters(
            img_np=img_np,
            all_object_data=[],
            cluster_labels=[],
            savename=cluster_vis_prefix,
            clean_sam_res=clean_masks,
            lvl1_clusters=lvl1_clusters,
            show=False,
        )
        LOGGER.info(f"Saved k-shot segmentation clusters: {cluster_vis_prefix}_clustered.png")

    LOGGER.info(f"Image: {len(clean_masks)} objects after stabilization.")
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    show_anns(clean_masks, borders=True)
    plt.axis("off")
    plt.savefig("sam2_segmentor.png")
    plt.show()
    plt.close()

    # 3. DINOv3 dense features
    dense_feats = _extract_dense_features(img_np, dino)

    # 4. Hierarchical Part Discovery
    parts, root = hpd.discover(clean_masks, dense_feats, img_np)

    LOGGER.info(f"HPD: {len(parts)} stable parts")

    return {
        "dense_feats": dense_feats,
        "parts": parts,
        "root": root,
        "sam_masks": clean_masks,
        "img_shape": (H, W),
    }


# ---------------------------------------------------------------------------
# Few-shot detector
# ---------------------------------------------------------------------------


class FewShotLogicalAnomalyDetector:
    """
    Few-shot logical anomaly detector using HS-CRL.

    Usage
    -----
    detector = LogicalAnomalyDetector(sam, stabilizer, dino, hpd, hscrl)
    detector.fit(normal_images)
    result = detector.predict(query_image)
    print(result["is_anomaly"], result["violations"])
    """

    def __init__(
        self,
        sam: SAM2Segmentor,
        stabilizer: MaskStabilizer,
        dino: DINOv3FeatureExtractor,
        hpd: HierarchicalPartDiscovery,
        hscrl: HSCRL,
    ):
        self.sam = sam
        self.stabilizer = stabilizer
        self.dino = dino
        self.hpd = hpd
        self.hscrl = hscrl

    # ------------------------------------------------------------------
    def fit(self, normal_images: List[np.ndarray]) -> None:
        """
        Register normal (anomaly-free) reference images.

        Internally:
          - processes each image (SAM -> DINOv3 -> HPD)
          - fits the part-type assigner (K-Means over all parts)
          - fills the patch memory bank
          - learns compositional rules P(child | parent)
          - calibrates the anomaly threshold (mu + k*sigma)
        """
        LOGGER.info(f"Fitting on {len(normal_images)} normal images ...")
        results = []
        for i, img in enumerate(normal_images):
            LOGGER.info(f"  Processing reference {i + 1}/{len(normal_images)}")
            results.append(
                process_one_image(
                    img,
                    self.sam,
                    self.stabilizer,
                    self.dino,
                    self.hpd,
                    cluster_vis_prefix=f"pipeline_output/segmentation/kshot/ref_{i:03d}",
                )
            )

        self.hscrl.fit(results)
        LOGGER.info("Fitting complete.")

    # ------------------------------------------------------------------
    def predict(
        self,
        query_image: np.ndarray,
        verbose: bool = False,
    ) -> Dict:
        """
        Score a query image.

        Returns
        -------
        dict with keys:
            is_anomaly         : bool
            total_energy       : float
            patch_energy       : float
            object_energy      : float
            composition_energy : float
            threshold          : float
            violations         : List[str]
            # heatmap            : np.ndarray  (only if return_heatmap=True)
            anomaly_map        : np.ndarray [H, W] float32, only present when
                                 is_anomaly=True.  Pixel value = mean absolute
                                 error of that patch against its part prototype.
            sam_masks          : List[Dict]  (for visualisation)
        """
        result = process_one_image(query_image, self.sam, self.stabilizer, self.dino, self.hpd)
        prediction = self.hscrl.predict(result)
        prediction["sam_masks"] = result["sam_masks"]
        prediction["parts"] = result["parts"]
        prediction["img_shape"] = result["img_shape"]

        if verbose:
            LOGGER.info(
                f"[PREDICT] anomaly={prediction['is_anomaly']} | "
                f"E={prediction['total_energy']:.4f} "
                f"(patch={prediction['patch_energy']:.4f}, "
                f"obj={prediction['object_energy']:.4f}, "
                f"comp={prediction['composition_energy']:.4f}) | "
                f"graph={prediction['graph_energy']:.4f}, "
                f"count={prediction['count_energy']:.4f}, "
                f"spatial={prediction['spatial_energy']:.4f}) | "
                f"threshold={prediction['threshold']:.4f}"
            )
            for v in prediction["violations"]:
                LOGGER.info(f"  -> {v}")

        return prediction


# ---------------------------------------------------------------------------
# Factory / convenience
# ---------------------------------------------------------------------------


def build_detector(
    sam_ckpt: str,
    sam_cfg: str,
    dino_model: str,
    device: str = "cpu",
    # HPD
    max_parts_per_region: int = 3,
    spatial_weight: float = 0.2,
    clustering_method: str = "agglomerative",
    contact_attach_thresh: float = 0.15,
    contact_near_thresh: float = 0.05,
    near_dist_thresh: float = 0.15,
    # HSCRL
    n_part_types: int = 32,
    n_object_types: int = 16,
    alpha: float = 0.10,
    beta: float = 0.15,
    gamma: float = 0.30,
    delta: float = 0.15,
    eta: float = 0.20,
    lam: float = 0.10,
    threshold_k: float = 2.5,
    presence_threshold: float = 0.2,
    top_k_fraction: float = 0.10,
) -> FewShotLogicalAnomalyDetector:
    """
    Convenience factory that assembles and returns a LogicalAnomalyDetector.

    Key sensitivity knobs
    ---------------------
    threshold_k      : higher → fewer false positives
    top_k_fraction   : fraction of worst patches used for E_patch (default 10%)
    presence_threshold: min P for a rule to be considered "required"
    """
    _device = torch.device(device)

    sam = SAM2Segmentor(model_cfg=sam_cfg, model_ckpt=sam_ckpt, device=_device)
    stabilizer = MaskStabilizer(min_area_ratio=0.002, max_elongation=15.0)
    dino = DINOv3FeatureExtractor(model_name=dino_model, device=_device)
    hpd = HierarchicalPartDiscovery(
        max_parts_per_region=max_parts_per_region,
        spatial_weight=spatial_weight,
        clustering_method=clustering_method,
        contact_attach_thresh=contact_attach_thresh,
        contact_near_thresh=contact_near_thresh,
        near_dist_thresh=near_dist_thresh,
    )
    hscrl = HSCRL(
        n_part_types=n_part_types,
        n_object_types=n_object_types,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        eta=eta,
        lam=lam,
        threshold_k=threshold_k,
        presence_threshold=presence_threshold,
        top_k_fraction=top_k_fraction,
    )

    return FewShotLogicalAnomalyDetector(sam, stabilizer, dino, hpd, hscrl)


def evaluate_test_set(
    detector: FewShotLogicalAnomalyDetector,
    test_dataset: MVTecLOCODataset,
    defects_config: str | Path,
    fpr_limit: float = 0.3,
) -> Dict:
    """
    Run inference on the full test split and compute MVTec LOCO metrics.

    Metrics computed
    ----------------
    image_auroc : ROC-AUC on image-level anomaly scores (threshold-free).
    spro        : Saturated Per-Region Overlap AUC (official localisation
                  metric), normalised to [0, fpr_limit].  Requires the
                  dataset to expose 'gt_mask' per sample and requires the
                  detector to return an 'anomaly_map' for anomalous images.
    accuracy    : Image-level accuracy at the Youden-optimal threshold.
    per_defect  : Per-defect-type breakdown (accuracy, precision, recall, F1).

    Parameters
    ----------
    detector       : fitted FewShotLogicalAnomalyDetector
    test_dataset   : MVTecLOCODataset (test split, all defect types)
    defects_config : path to the category's defects_config.json
    fpr_limit      : upper FPR limit for sPRO AUC normalisation (default 0.3)
    """
    if len(test_dataset) == 0:
        raise RuntimeError("Test split is empty; cannot evaluate.")

    evaluator = LocoEvaluator.from_config(defects_config, fpr_limit=fpr_limit)
    reports: List[Dict] = []
    n_total = len(test_dataset)

    for i in range(n_total):
        sample = test_dataset[i]
        img_np = _sample_to_u8(sample)
        defect_type = str(sample.get("defect_type", "good"))
        img_path = str(sample.get("img_path", f"index_{i}"))

        # GT mask: uint8 numpy array from dataset, or None for good images.
        # MVTec LOCO datasets expose the raw GT mask as 'mask' or 'gt_mask'.
        gt_mask: Optional[np.ndarray] = None
        for key in ("gt_mask", "mask"):
            raw = sample.get(key)
            if raw is not None:
                if isinstance(raw, torch.Tensor):
                    gt_mask = raw.squeeze().cpu().numpy().astype(np.uint8)
                else:
                    gt_mask = np.asarray(raw).astype(np.uint8)
                break

        report = detector.predict(img_np, verbose=False)
        anomaly_map = report.get("anomaly_map")  # float32 [H, W] or None
        score = float(report.get("total_energy", 0.0))

        # Resize anomaly_map and gt_mask to the same spatial size if needed
        if anomaly_map is not None and gt_mask is not None:
            if anomaly_map.shape != gt_mask.shape:
                import cv2 as _cv2

                gt_mask = _cv2.resize(
                    gt_mask,
                    (anomaly_map.shape[1], anomaly_map.shape[0]),
                    interpolation=_cv2.INTER_NEAREST,
                )

        evaluator.add(
            anomaly_score=score,
            gt_mask=gt_mask,
            anomaly_map=anomaly_map,
            defect_type=defect_type,
            img_path=img_path,
        )

        reports.append(
            {
                "index": i,
                "img_path": img_path,
                "defect_type": defect_type,
                "gt_label": 0 if (gt_mask is None or not np.any(gt_mask > 0)) else 1,
                "score": score,
                "report": report,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            LOGGER.info(f"Evaluated {i + 1}/{n_total} test images")

    metrics = evaluator.compute()
    metrics["reports"] = reports
    return metrics


def _sample_to_u8(sample: Dict) -> np.ndarray:
    img = sample["image"]
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.asarray(img)

    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_heatmap_outputs(query_image: np.ndarray, report: Dict, prefix: str) -> None:
    """Save optional raw heatmap and overlay if present in prediction report."""
    heatmap = report.get("heatmap")
    if heatmap is None:
        return

    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heatmap_dir = Path("pipeline_output/heatmaps")
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap_norm, cmap="inferno")
    plt.colorbar(label="Prototype anomaly score")
    plt.title("Prototype Anomaly Heatmap")
    plt.axis("off")
    plt.tight_layout()
    raw_heatmap_path = heatmap_dir / f"{prefix}_heatmap.png"
    plt.savefig(raw_heatmap_path, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(query_image)
    plt.imshow(heatmap_norm, cmap="inferno", alpha=0.45)
    plt.title("Anomaly Heatmap Overlay")
    plt.axis("off")
    plt.tight_layout()
    overlay_path = heatmap_dir / f"{prefix}_overlay.png"
    plt.savefig(overlay_path, dpi=150)
    plt.close()

    LOGGER.info(f"Saved anomaly heatmap: {raw_heatmap_path}")
    LOGGER.info(f"Saved anomaly overlay: {overlay_path}")


def main():
    """Train a few-shot reference bank and evaluate on the full test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((1024, 1024), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    train_dataset = MVTecLOCODataset(
        "/home/yijin/projects/logic_agent/data",
        "splicing_connectors",
        subset="train",
        img_size=1024,
        transform=transform,
    )
    test_dataset = MVTecLOCODataset(
        "/home/yijin/projects/logic_agent/data",
        "splicing_connectors",
        subset="test",
        anomaly_type=None,
        img_size=1024,
        transform=transform,
    )

    # Build reference set from normal train images
    n_refs = min(10, len(train_dataset))
    reference_images: List[np.ndarray] = []
    for i in range(n_refs):
        reference_images.append(_sample_to_u8(train_dataset[i]))

    if not reference_images:
        raise RuntimeError("No reference images available in train split.")

    LOGGER.info(f"Using device: {device}")
    LOGGER.info(f"Reference images: {len(reference_images)}")
    LOGGER.info(f"Test samples: {len(test_dataset)}")

    detector = build_detector(
        sam_ckpt=SAM2_CKPT,
        sam_cfg=MODEL_CFG,
        dino_model=DINO_MODEL,
        device=str(device),
        max_parts_per_region=4,
        clustering_method="agglomerative",
    )

    LOGGER.info("Fitting detector on reference images...")
    detector.fit(reference_images)

    LOGGER.info("Running full-test inference and evaluation...")
    defects_cfg_path = Path("/home/yijin/projects/logic_agent/data") / "splicing_connectors" / "defects_config.json"
    metrics = evaluate_test_set(detector, test_dataset, defects_config=defects_cfg_path)

    LOGGER.info("================ Test Evaluation ================")
    LOGGER.info(f"Samples total : {metrics['n_total']}  (good={metrics['n_good']}, anomaly={metrics['n_anomaly']})")
    LOGGER.info(f"Image AUROC   : {metrics['image_auroc']:.4f}")
    LOGGER.info(f"sPRO (AUC@{0.3:.0%} FPR) : {metrics['spro']:.4f}")
    LOGGER.info(f"Accuracy      : {metrics['accuracy']:.4f}  (threshold={metrics['optimal_threshold']:.4f})")
    LOGGER.info(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    LOGGER.info(f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    # Log mean energy breakdown over anomalous images
    anom_reports = [r for r in metrics.get("reports", []) if r["gt_label"] == 1]
    if anom_reports:

        def _mean_e(key):
            vals = [r["report"].get(key, 0.0) for r in anom_reports]
            return sum(vals) / len(vals)

        LOGGER.info(
            f"Mean energies on anomalous images: "
            f"patch={_mean_e('patch_energy'):.4f} "
            f"obj={_mean_e('object_energy'):.4f} "
            f"comp={_mean_e('composition_energy'):.4f} "
            f"graph={_mean_e('graph_energy'):.4f} "
            f"count={_mean_e('count_energy'):.4f} "
            f"spatial={_mean_e('spatial_energy'):.4f}"
        )
    LOGGER.info("Per-defect breakdown:")
    for defect_type, dm in metrics["per_defect"].items():
        LOGGER.info(
            f"  {defect_type:40s}  acc={dm['accuracy']:.3f}  f1={dm['f1']:.3f}  (n={dm['n']}, anom={dm['n_anomaly']})"
        )
    reports = metrics["reports"]
    if not reports:
        return

    representative = next((r for r in reports if r["gt_label"] == 1), reports[0])
    representative_sample = test_dataset[representative["index"]]
    representative_image = _sample_to_u8(representative_sample)
    representative_report = representative["report"]
    representative_path = representative["img_path"]
    representative_type = representative["defect_type"]

    LOGGER.info("Saving visualizations for one representative test sample")
    LOGGER.info(f"Sample: {representative_path} (defect_type={representative_type})")

    save_heatmap_outputs(representative_image, representative_report, prefix="representative_query")
    visualize_segmentation_results(
        image=representative_image,
        sam_masks=representative_report.get("sam_masks", []),
        save_path="pipeline_output/segmentation/representative_query_segmentation_overlay.png",
        show=False,
    )


if __name__ == "__main__":
    main()
