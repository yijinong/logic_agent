"""
Top-Down Semantic Segmentation Pipeline
========================================

Hierarchy
---------
Level 0  (Global Analysis)   : K-Means / HDBSCAN on DINOv3 patch features
                                → discovers K semantic concepts in the image.
Level 1  (Prototype Query)   : Each cluster centroid becomes a Query Prototype.
Level 2  (SAM2 Prompting)    : Cosine-similarity map between prototype and
                                dense features → peak locations → SAM2 point
                                prompts → crisp object masks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel
from transformers.models.dinov3_vit import DINOv3ViTPreTrainedModel

from logic_agent.dataset.mvtec_loco import MVTecLOCODataset
from logic_agent.logging import get_logger
from logic_agent.model.segment import (
    SAM2PointSegmentor,
    SimilarityMapPrompter,
    visualize_topdown_results,
)

DINO_MODEL = "/data/LLM_Weights/dinov3/dinov3-vitl16-pretrain-lvd1689m"
SAM2_CKPT = "/data/LLM_Weights/sam2.1-hiera-large/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# DINOv3 Feature Extractor (unchanged from original)
# ---------------------------------------------------------------------------


class DINOv3FeatureExtractor(nn.Module):
    """
    DINOv3-based Feature Extractor for Multi-Scale dense feature extraction with SAM masks
    """

    def __init__(
        self,
        model_name: str = DINO_MODEL,
        unfreeze_layers: list[int] = [20, 21, 22, 23],
        unfreeze_norm: bool = True,
        num_protos_per_obj: int = 5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model, self.processor = self._load_model(model_name)
        self.device = device
        self.model = self.model.to(device)
        self._freeze_backbone(unfreeze_layers, unfreeze_norm)

        self.hidden_size = self.model.config.hidden_size
        self.layer_indices = [4, 8, 12, 16]
        self.num_protos_per_obj = num_protos_per_obj
        self.attn_layer_index = -1

    def _load_model(self, model_name):
        LOGGER.info(f"Loading DINOv3 model from {model_name}")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model: DINOv3ViTPreTrainedModel = AutoModel.from_pretrained(model_name)
        model.set_attn_implementation("eager")
        LOGGER.info(
            f"DINOv3 model loaded with "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} "
            f"trainable parameters"
        )
        return model, processor

    def _freeze_backbone(self, layers, unfreeze_norm=True):
        for name, param in self.model.named_parameters():
            layer_match = any(
                (f"encoder.layer.{layer_idx}." in name or f"layer.{layer_idx}." in name) for layer_idx in layers
            )
            if layer_match:
                param.requires_grad = True
            elif unfreeze_norm and "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _prepare_mask(self, sam_mask, grid_size):
        B = sam_mask.shape[0]
        mask_weights = F.interpolate(sam_mask.float(), size=(grid_size, grid_size), mode="area")
        geo_weights = mask_weights.reshape(B, -1, 1)
        return geo_weights

    def _get_cls_attn(self, outputs):
        attn = outputs.attentions[-1]
        attn = attn.mean(dim=1)
        cls_attn = attn[:, 0, :]
        num_registers = getattr(self.model.config, "num_register_tokens", 0)
        patch_start_idx = 1 + num_registers
        patch_attn = cls_attn[:, patch_start_idx:]
        return patch_attn.unsqueeze(-1)

    def _compute_multi_scale_feats(self, outputs):
        all_feats = []
        for idx in self.layer_indices:
            layer_output = outputs.hidden_states[idx]
            patches = layer_output[:, 1 + self.model.config.num_register_tokens :, :]
            all_feats.append(patches)
        multi_scale_feats = torch.cat(all_feats, dim=-1)
        return F.normalize(multi_scale_feats, p=2, dim=-1)

    @torch.no_grad()
    def extract_dense_features(self, x):
        """
        Extract multi-scale dense feature maps and attention maps.

        Args:
            x: [B, C, H, W]

        Returns:
            multi_scale_feats : [B, N, D_total]
            cls_attn          : [B, N, 1]
            global_cls        : [B, D_total]
        """
        outputs = self.model(x, output_hidden_states=True, output_attentions=True)
        multi_scale_feats = self._compute_multi_scale_feats(outputs)
        cls_attn = self._get_cls_attn(outputs)
        global_cls = outputs.last_hidden_state[:, 0, :]
        global_cls = F.normalize(global_cls, p=2, dim=-1)

        LOGGER.debug(f"Multi-scale feature shape: {multi_scale_feats.shape}")
        LOGGER.debug(f"CLS attention shape: {cls_attn.shape}")
        LOGGER.debug(f"Global CLS feature shape: {global_cls.shape}")

        return multi_scale_feats, cls_attn, global_cls

    def pool_object_prototype(self, multi_scale_feats, cls_attn, sam_mask):
        """
        Pool object prototype using semantic-geometric fusion.

        Args:
            multi_scale_feats : [B, N, D_total]
            cls_attn          : [B, N, 1]
            sam_mask          : [B, 1, H, W]

        Returns:
            prototype : [B, 1, D_total]
        """
        B, N, D_total = multi_scale_feats.shape
        grid_size = int(N**0.5)

        geo_weights = self._prepare_mask(sam_mask, grid_size)
        sem_weights = cls_attn
        combined_weights = geo_weights * sem_weights
        sum_weights = torch.clamp(combined_weights.sum(dim=1, keepdim=True), min=1e-6)
        norm_weights = combined_weights / sum_weights

        prototype = (multi_scale_feats * norm_weights).sum(dim=1, keepdim=True)
        prototype = F.normalize(prototype, p=2, dim=-1)

        LOGGER.debug(f"Pooled prototype shape: {prototype.shape}")
        return prototype

    def compute_semantic_divergence(self, child_proto, parent_proto):
        return 1 - F.cosine_similarity(child_proto, parent_proto, dim=-1)

    @torch.no_grad()
    def forward(self, x, mask):
        multi_scale_feats, cls_attn, global_cls = self.extract_dense_features(x)
        prototype = self.pool_object_prototype(multi_scale_feats, cls_attn, mask)
        return multi_scale_feats, prototype, global_cls


# ---------------------------------------------------------------------------
# NEW: Level 0 — Global Semantic Clusterer
# ---------------------------------------------------------------------------


class GlobalSemanticClusterer:
    """
    Level 0 of the Top-Down hierarchy.

    Performs global clustering of all DINOv3 patch features to discover the
    main semantic concepts present in the image (sky, object type, background,
    etc.).  Each cluster centroid becomes a **Query Prototype** that is used
    to interrogate SAM2 in the subsequent levels.

    Supported methods
    -----------------
    ``"kmeans"``  : Fast, deterministic number of concepts.  Preferred default.
    ``"hdbscan"`` : Data-driven concept count.  Requires the ``hdbscan``
                    package.  Useful when ``n_concepts`` is unknown in advance.

    Args:
        n_concepts      : Number of clusters for K-Means (ignored for HDBSCAN).
        method          : ``"kmeans"`` or ``"hdbscan"``.
        min_cluster_size: Minimum patches per cluster for HDBSCAN.
        random_state    : Reproducibility seed.
    """

    def __init__(
        self,
        n_concepts: int = 8,
        method: str = "kmeans",
        min_cluster_size: int = 15,
        random_state: int = 42,
    ):
        self.n_concepts = n_concepts
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(self, dense_feats: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        """
        Assign every patch to a semantic concept and compute concept prototypes.

        Args:
            dense_feats: [N, D] L2-normalised patch features for a **single**
                         image (no batch dimension).

        Returns:
            patch_labels  : [N] int array — concept id for every patch.
            prototypes    : [K, D] L2-normalised cluster centroids (Tensor).
                            K = n_concepts for K-Means; data-driven for HDBSCAN.
        """
        feats_np = dense_feats.detach().cpu().numpy()  # [N, D]

        if self.method == "kmeans":
            patch_labels, proto_np = self._run_kmeans(feats_np)
        elif self.method == "hdbscan":
            patch_labels, proto_np = self._run_hdbscan(feats_np)
        else:
            raise ValueError(f"Unknown clustering method: '{self.method}'. Choose 'kmeans' or 'hdbscan'.")

        proto_tensor = F.normalize(torch.from_numpy(proto_np).float(), p=2, dim=-1)  # [K, D]

        LOGGER.info(
            f"GlobalSemanticClusterer ({self.method}): "
            f"{proto_tensor.shape[0]} concepts discovered from {len(patch_labels)} patches."
        )

        return patch_labels, proto_tensor

    # ------------------------------------------------------------------
    # Internal clustering strategies
    # ------------------------------------------------------------------

    def _run_kmeans(self, feats_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Standard K-Means on the full patch feature matrix.

        Cluster centres are returned as raw (un-normalised) centroids;
        L2-normalisation is applied in ``cluster()``.
        """
        km = KMeans(
            n_clusters=self.n_concepts,
            n_init=10,
            max_iter=300,
            random_state=self.random_state,
        )
        labels = km.fit_predict(feats_np)  # [N]
        centroids = km.cluster_centers_  # [K, D]
        return labels, centroids

    def _run_hdbscan(self, feats_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Density-based clustering — automatically determines the number of
        concepts.  Noise patches (label == -1) are assigned to the nearest
        cluster centroid as a post-processing step.
        """
        try:
            import hdbscan
        except ImportError as e:
            raise ImportError("hdbscan is required for method='hdbscan'.  Install it with: pip install hdbscan") from e

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels: np.ndarray = clusterer.fit_predict(feats_np)  # [N], -1 = noise

        valid_labels = sorted(set(labels[labels >= 0]))
        if not valid_labels:
            # Edge case: everything is noise — fall back to K-Means
            LOGGER.warning("HDBSCAN produced only noise.  Falling back to K-Means.")
            return self._run_kmeans(feats_np)

        centroids = np.stack([feats_np[labels == lbl].mean(axis=0) for lbl in valid_labels])  # [K, D]

        # Re-assign noise patches to the nearest centroid (cosine distance)
        if (labels == -1).any():
            noise_mask = labels == -1
            noise_feats = feats_np[noise_mask]  # [M, D]
            # cosine similarity = dot product (features are L2-normalised)
            sims = noise_feats @ centroids.T  # [M, K]
            labels[noise_mask] = sims.argmax(axis=1)

        return labels, centroids


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def visualize_clusters(
    img_np: np.ndarray,
    all_object_data: List[Dict[str, Any]],
    cluster_labels: List[int],
    savename: str,
    clean_sam_res: Optional[List[Dict[str, Any]]] = None,
    lvl1_clusters: Optional[Dict[int, List[Any]]] = None,
    show: bool = False,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(img_np)
    axes[1].set_title("Clustered Overlay")
    axes[1].axis("off")

    cmap = plt.get_cmap("tab10")

    if clean_sam_res is not None:
        for i, mask_dict in enumerate(clean_sam_res):
            mask = mask_dict.get("segmentation") if isinstance(mask_dict, dict) else None
            if mask is None:
                continue
            color = cmap(i % 10)[:3]
            overlay = np.zeros((*mask.shape, 4))
            mask_np = mask.astype(bool)
            overlay[mask_np] = [*color, 0.35]
            axes[0].imshow(overlay)
            ys, xs = np.where(mask_np)
            if len(xs):
                axes[0].text(
                    int(xs.mean()),
                    int(ys.mean()),
                    str(i),
                    color="white",
                    fontsize=9,
                    ha="center",
                    va="center",
                    weight="bold",
                    bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
                )

    if lvl1_clusters is not None:
        for lbl, members in lvl1_clusters.items():
            color = cmap(int(lbl) % 10)[:3]
            for member in members:
                mask = None
                if isinstance(member, dict):
                    for k in ("segmentation", "mask", "sam_mask"):
                        candidate = member.get(k)
                        if candidate is not None:
                            mask = candidate
                            break
                else:
                    for attr in ("sam_mask", "mask"):
                        candidate = getattr(member, attr, None)
                        if candidate is not None:
                            mask = candidate
                            break
                if mask is None:
                    continue
                mask_np = mask.cpu().numpy().astype(bool) if isinstance(mask, torch.Tensor) else mask.astype(bool)
                overlay = np.zeros((*mask_np.shape, 4))
                overlay[mask_np] = [*color, 0.5]
                axes[1].imshow(overlay)
                ys, xs = np.where(mask_np)
                if len(xs):
                    axes[1].text(
                        int(xs.mean()),
                        int(ys.mean()),
                        str(lbl),
                        color="white",
                        fontsize=10,
                        ha="center",
                        va="center",
                        weight="bold",
                        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
                    )
    else:
        for i, obj in enumerate(all_object_data):
            mask = obj.get("mask") if isinstance(obj, dict) else getattr(obj, "mask", None)
            label = cluster_labels[i]
            if mask is None or label == -1:
                continue
            color = cmap(label % 10)[:3]
            mask_np = mask.cpu().numpy().astype(bool) if isinstance(mask, torch.Tensor) else mask.astype(bool)
            overlay = np.zeros((*mask_np.shape, 4))
            overlay[mask_np] = [*color, 0.5]
            axes[1].imshow(overlay)

    plt.tight_layout()
    out_path = Path(f"{savename}_clustered.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close()


def save_segmented_masks(
    concept_masks: dict[int, list[dict]],
    image_shape: tuple[int, int],
    save_dir: str | Path,
    prefix: str,
) -> dict[str, Any]:
    """Save per-concept binary masks and a merged concept map.

    Args:
        concept_masks: Output from top-down pipeline, keyed by concept id.
        image_shape: (H, W) of the source image.
        save_dir: Directory where mask files and metadata are written.
        prefix: Filename prefix (for example ``topdown_0``).

    Returns:
        Summary dictionary with saved file paths and counts.
    """
    H, W = image_shape
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_map = np.zeros((H, W), dtype=np.int32)
    saved_masks: list[dict[str, Any]] = []

    for cid, masks in concept_masks.items():
        for midx, m_dict in enumerate(masks):
            mask = m_dict.get("segmentation")
            if mask is None:
                continue

            mask_bool = mask.astype(bool)
            mask_u8 = mask_bool.astype(np.uint8) * 255
            mask_path = out_dir / f"{prefix}_concept_{cid}_mask_{midx}.png"
            plt.imsave(mask_path, mask_u8, cmap="gray", vmin=0, vmax=255)

            # Concept ids are stored as cid+1 so that 0 remains background.
            concept_map[mask_bool] = int(cid) + 1

            saved_masks.append(
                {
                    "concept_id": int(cid),
                    "mask_index": int(midx),
                    "path": str(mask_path),
                    "area": int(mask_bool.sum()),
                    "score": float(m_dict.get("score", 0.0)),
                    "num_prompts": int(len(m_dict.get("prompt_points", []))),
                }
            )

    concept_map_path = out_dir / f"{prefix}_concept_map.npy"
    np.save(concept_map_path, concept_map)

    preview_path = out_dir / f"{prefix}_concept_map.png"
    plt.imsave(preview_path, concept_map, cmap="tab20")

    metadata = {
        "prefix": prefix,
        "image_shape": [int(H), int(W)],
        "num_concepts": int(len(concept_masks)),
        "num_saved_masks": int(len(saved_masks)),
        "concept_map_npy": str(concept_map_path),
        "concept_map_preview": str(preview_path),
        "masks": saved_masks,
    }
    metadata_path = out_dir / f"{prefix}_masks_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info(f"Saved {len(saved_masks)} masks and merged concept map to {out_dir} (metadata: {metadata_path.name})")

    return metadata


# ---------------------------------------------------------------------------
# Main: Top-Down Pipeline
# ---------------------------------------------------------------------------


def main():
    """
    Top-Down Semantic Segmentation Pipeline
    ----------------------------------------
    Level 0  Extract DINOv3 dense features for the full image.
             Cluster all N patches with K-Means → K concept prototypes.

    Level 1  Each cluster centroid is a Query Prototype.

    Level 2  For each prototype:
               • Compute cosine similarity map over the patch grid.
               • Find spatial peaks via NMS.
               • Convert peaks to image-space (x, y) points.
               • Feed those points as prompts to SAM2ImagePredictor.
               • Receive crisp object masks back.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((1024, 1024), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = MVTecLOCODataset(
        "/home/yijin/projects/logic_agent/data",
        "splicing_connectors",
        subset="train",
        img_size=1024,
        transform=transform,
    )

    # ----------------------------------------------------------------
    # Model initialisation
    # ----------------------------------------------------------------
    LOGGER.info("Loading DINOv3 feature extractor …")
    dino_model = DINOv3FeatureExtractor(DINO_MODEL, device=device)
    dino_model.eval()

    LOGGER.info("Loading SAM2 point-prompted segmentor …")
    sam2_model = SAM2PointSegmentor(
        model_cfg=SAM2_CFG,
        model_ckpt=SAM2_CKPT,
        device=device,
        score_threshold=0.50,
    )

    # ----------------------------------------------------------------
    # Hyperparameters
    # ----------------------------------------------------------------
    N_CONCEPTS = 8  # Level 0: number of K-Means clusters
    N_PEAKS = 5  # Level 2: max SAM2 point prompts per concept
    SIM_THRESH = 0.35  # Level 2: minimum cosine similarity for a peak
    NMS_RADIUS = 3  # Level 2: NMS window half-size (in patch units)

    clusterer = GlobalSemanticClusterer(
        n_concepts=N_CONCEPTS,
        method="hdbscan",
    )
    prompter = SimilarityMapPrompter(
        n_peaks=N_PEAKS,
        sim_threshold=SIM_THRESH,
        nms_radius=NMS_RADIUS,
    )

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    with torch.inference_mode():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            # ---- 0. Prepare image ----
            img_item = sample["image"]
            if isinstance(img_item, torch.Tensor):
                img_t = img_item[0] if img_item.ndim == 4 else img_item
                img_np = img_t.cpu().numpy().transpose(1, 2, 0) if img_t.shape[0] in (1, 3) else img_t.cpu().numpy()
            else:
                img_np = np.array(img_item)

            img_float = img_np.astype(np.float32)
            if img_float.max() > 1.0:
                img_float /= 255.0
            img_u8 = (img_float * 255).clip(0, 255).astype(np.uint8)

            H, W = img_u8.shape[:2]

            # ---- 1. DINOv3 forward pass ----
            dino_inputs = dino_model.processor(images=img_float, return_tensors="pt")
            dino_inputs["pixel_values"] = dino_inputs["pixel_values"].to(device)

            multi_scale_feats, cls_attn, global_cls = dino_model.extract_dense_features(dino_inputs["pixel_values"])
            # Remove batch dim → [N, D]
            patch_feats: torch.Tensor = multi_scale_feats.squeeze(0)
            LOGGER.info(f"[{idx}] Patch features: {patch_feats.shape}")

            # ---- 2. Level 0 — Global Clustering ----
            patch_labels, concept_prototypes = clusterer.cluster(patch_feats)
            # patch_labels     : [N]  — concept id per patch
            # concept_prototypes: [K, D] — L2-normed cluster centroids
            K = concept_prototypes.shape[0]
            LOGGER.info(f"[{idx}] Level 0: {K} concepts discovered.")

            # ---- 3. Level 1 + 2 — Prototype → Peaks → SAM2 ----
            concept_masks: dict[int, list[dict]] = {}
            sim_maps: dict[int, np.ndarray] = {}

            # Pre-encode image once; all concept queries reuse the same
            # SAM2 image embedding (avoids re-encoding K times).
            sam2_model.set_image(img_u8)

            for cid in range(K):
                prototype: torch.Tensor = concept_prototypes[cid]  # [D]
                # Ensure prototype is on same device as patch features
                prototype = prototype.to(patch_feats.device)

                # Level 1: similarity map + peak extraction
                point_coords, sim_map = prompter.get_point_prompts(
                    patch_feats,
                    prototype,
                    image_size=H,  # assumes square images
                )
                sim_maps[cid] = sim_map

                if len(point_coords) == 0:
                    LOGGER.debug(f"[{idx}] Concept {cid}: no peaks above threshold, skipping.")
                    concept_masks[cid] = []
                    continue

                # Level 2: SAM2 point-prompted segmentation
                #
                # Strategy: prompt SAM2 once with ALL peaks for this concept.
                # This lets SAM2 jointly reason about multiple foreground hints,
                # which is better than independent single-point calls when the
                # concept spans multiple disconnected regions.
                point_labels = np.ones(len(point_coords), dtype=np.int32)  # all foreground
                mask, score = sam2_model.segment_with_points(point_coords, point_labels)

                masks_for_concept = []
                if mask is not None:
                    area = int(mask.sum())
                    area_ratio = area / float(H * W)

                    masks_for_concept.append(
                        {
                            "segmentation": mask,
                            "area": area,
                            "score": score,
                            "concept_id": cid,
                            "prompt_points": point_coords,
                            "area_ratio": area_ratio,
                        }
                    )
                    LOGGER.info(
                        f"[{idx}] Concept {cid}: mask area={area}, area_ratio={area_ratio:.3f}, "
                        f"score={score:.3f}, n_prompts={len(point_coords)}"
                    )
                else:
                    LOGGER.debug(f"[{idx}] Concept {cid}: SAM2 score below threshold, no mask.")

                concept_masks[cid] = masks_for_concept

            # ---- 4. Visualise ----
            visualize_topdown_results(
                img_np=img_u8,
                concept_masks=concept_masks,
                sim_maps=sim_maps,
                patch_labels=patch_labels,
                savename=f"topdown_{idx}",
                show=False,
            )

            # ---- 5. Save binary masks and merged concept map ----
            save_segmented_masks(
                concept_masks=concept_masks,
                image_shape=(H, W),
                save_dir="pipeline_output/topdown_masks",
                prefix=f"topdown_{idx}",
            )

            LOGGER.info(f"[{idx}] Done. Processed {sum(len(v) for v in concept_masks.values())} concept masks.")

            # Process only the first image for demonstration
            break

    # ---- Cleanup ----
    try:
        del sam2_model, dino_model
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
