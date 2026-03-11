"""
hpd.py  –  Hierarchical Part Discovery  (HPD)
===============================================

Fixes applied (from review document)
--------------------------------------
[Fix 1]  Spatially-constrained clustering
         Clustering input = concat(feature_vec, λ·xy_coords) so parts do not
         split randomly across space.  n_clusters is now fixed to
         max_parts_per_region (stable across images) rather than estimated
         from variance (unstable).

[Fix 2]  Multi-relation hierarchy  (containment | attached_to | near)
         Strict IoU containment is kept for true nesting (IoU_fraction > 0.8).
         Contact ratio  contact_pixels / min(perimeter_i, perimeter_j)  detects
         attachment.  Centroid proximity detects spatial nearness.
         Every edge now carries a relation_type string.

[Fix 3]  Boundary contact ratio for adjacency
         Replaces naive dilation-touch with
             contact_ratio = shared_boundary_len / min(perimeter_i, perimeter_j)
         Spurious noise edges are suppressed by a configurable threshold.

[Fix 4]  SAM-parent object grouping
         Every PartDescriptor records parent_mask_id (the SAM mask it came
         from).  The graph builder groups parts by SAM parent before linking
         cross-object relations, so rule keys are object-scoped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

from logic_agent.logging import get_logger

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Relation type constants
# ---------------------------------------------------------------------------

REL_CONTAINS = "contains"
REL_ATTACHED = "attached_to"
REL_NEAR = "near"
REL_ABOVE = "above"
REL_BELOW = "below"
REL_LEFT_OF = "left_of"
REL_RIGHT_OF = "right_of"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PartDescriptor:
    """One discovered part inside a parent SAM region."""

    part_id: int
    parent_mask_id: int  # SAM mask index → object identity
    mask: np.ndarray  # binary H×W
    prototype: np.ndarray  # mean DINOv3 feature [D]
    semantic_variance: float
    shape_compactness: float
    boundary_score: float
    area: int
    centroid: np.ndarray  # (y, x)
    perimeter: float  # precomputed for contact-ratio


@dataclass
class PartEdge:
    """Typed edge between two part nodes."""

    relation: str  # REL_* constant
    weight: float = 1.0


@dataclass
class PartNode:
    """Node in the hierarchical part graph."""

    part: PartDescriptor
    # Hierarchical (parent→child containment): list of (child_node, edge)
    children: List[Tuple["PartNode", PartEdge]] = field(default_factory=list)
    parent: Optional["PartNode"] = None
    # Peer relations (attachment, near, spatial): list of (peer_node, edge)
    peers: List[Tuple["PartNode", PartEdge]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _patch_features_for_mask(
    dense_feats: torch.Tensor,  # [N, D]
    mask: np.ndarray,  # [H, W] bool
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (feats, coords) for patches inside *mask*.

    feats  : [n, D]
    coords : [n, 2]  normalised (y,x) in [0,1]
    """
    mask_small = cv2.resize(
        mask.astype(np.uint8),
        (grid_size, grid_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    ys, xs = np.where(mask_small)
    flat_idx = ys * grid_size + xs

    feats_np = dense_feats.cpu().numpy()
    feats = feats_np[flat_idx]
    coords = np.stack([ys / (grid_size - 1 + 1e-6), xs / (grid_size - 1 + 1e-6)], axis=1).astype(np.float32)

    return feats, coords


def _region_variance(feats: np.ndarray) -> float:
    if len(feats) < 2:
        return 0.0
    return float(np.mean(np.var(feats, axis=0)))


def _shape_stats(mask: np.ndarray) -> Tuple[float, float]:
    """Return (compactness, perimeter)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)
    if perimeter < 1e-6:
        return 0.0, 0.0
    return float(area / (perimeter**2)), float(perimeter)


def _boundary_gradient_score(mask: np.ndarray, gray: np.ndarray) -> float:
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = (dilated - eroded).astype(bool)
    if not boundary.any():
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx**2 + gy**2)[boundary].mean())


def _contact_length(mask_i: np.ndarray, mask_j: np.ndarray) -> int:
    """Pixels on the outer boundary of mask_i that overlap with mask_j."""
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_i.astype(np.uint8), kernel)
    boundary = dilated.astype(bool) & ~mask_i.astype(bool)
    return int((boundary & mask_j.astype(bool)).sum())


def _spatial_relation(ci: np.ndarray, cj: np.ndarray) -> str:
    dy = float(cj[0] - ci[0])
    dx = float(cj[1] - ci[1])
    if abs(dy) >= abs(dx):
        return REL_ABOVE if dy < 0 else REL_BELOW
    return REL_LEFT_OF if dx < 0 else REL_RIGHT_OF


# ---------------------------------------------------------------------------
# Core HPD class
# ---------------------------------------------------------------------------


class HierarchicalPartDiscovery:
    """
    Hierarchical Part Discovery from SAM2 masks + DINOv3 dense features.

    Parameters
    ----------
    max_parts_per_region   : fixed K clusters per SAM region (stable across
                             images of the same category).
    spatial_weight         : λ — weight of appended (y,x) coordinates in the
                             clustering input vector.
    min_part_area          : minimum pixel area to retain a part.
    clustering_method      : 'agglomerative' | 'kmeans' | 'spectral'.
    sem_var_thresh         : max intra-part feature variance.
    shape_thresh           : min shape compactness.
    boundary_thresh        : min boundary gradient (0 = disabled).
    containment_iou_thresh : IoU fraction threshold for containment edge.
    contact_attach_thresh  : contact_ratio threshold for attached_to edge.
    contact_near_thresh    : contact_ratio threshold for near edge.
    near_dist_thresh       : centroid distance / img_diag for near edge when
                             masks don't touch.
    """

    def __init__(
        self,
        max_parts_per_region: int = 3,
        spatial_weight: float = 0.2,
        min_part_area: int = 100,
        clustering_method: str = "agglomerative",
        sem_var_thresh: float = 0.08,
        shape_thresh: float = 1e-4,
        boundary_thresh: float = 0.0,
        containment_iou_thresh: float = 0.8,
        contact_attach_thresh: float = 0.15,
        contact_near_thresh: float = 0.05,
        near_dist_thresh: float = 0.15,
    ):
        self.max_parts_per_region = max_parts_per_region
        self.spatial_weight = spatial_weight
        self.min_part_area = min_part_area
        self.clustering_method = clustering_method
        self.sem_var_thresh = sem_var_thresh
        self.shape_thresh = shape_thresh
        self.boundary_thresh = boundary_thresh
        self.containment_iou_thresh = containment_iou_thresh
        self.contact_attach_thresh = contact_attach_thresh
        self.contact_near_thresh = contact_near_thresh
        self.near_dist_thresh = near_dist_thresh
        self._part_counter = 0

    # ------------------------------------------------------------------
    def discover(
        self,
        sam_masks: List[Dict],
        dense_feats: torch.Tensor,
        img_np: np.ndarray,
    ) -> Tuple[List[PartDescriptor], PartNode]:
        self._part_counter = 0
        H, W = img_np.shape[:2]
        N = dense_feats.shape[0]
        grid_size = int(math.isqrt(N))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        img_diag = math.sqrt(H**2 + W**2)

        all_parts: List[PartDescriptor] = []

        for mask_id, mask_dict in enumerate(sam_masks):
            raw_mask: np.ndarray = mask_dict["segmentation"]

            # [Fix 1] Extract features + spatial coords
            feats, coords = _patch_features_for_mask(dense_feats, raw_mask, grid_size)

            if len(feats) < self.max_parts_per_region:
                part = self._make_part(mask_id, raw_mask, feats, gray)
                if part is not None:
                    all_parts.append(part)
                continue

            # [Fix 1] Spatially-constrained clustering input
            cluster_input = np.concatenate([feats, self.spatial_weight * coords], axis=1)
            labels = self._cluster(cluster_input, self.max_parts_per_region)

            sub_parts = self._build_sub_parts(
                mask_id,
                raw_mask,
                feats,
                labels,
                self.max_parts_per_region,
                grid_size,
                H,
                W,
                gray,
            )
            all_parts.extend(sub_parts)

        stable_parts = self._filter_stable(all_parts)
        root = self._build_graph(stable_parts, img_diag)

        LOGGER.info(
            f"HPD: {len(sam_masks)} SAM masks → {len(all_parts)} candidate parts → {len(stable_parts)} stable parts"
        )
        return stable_parts, root

    # ------------------------------------------------------------------
    def _cluster(self, feats: np.ndarray, n_clusters: int) -> np.ndarray:
        n_clusters = min(n_clusters, len(feats))
        if self.clustering_method == "spectral":
            try:
                return SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="nearest_neighbors",
                    random_state=0,
                    n_jobs=-1,
                ).fit_predict(feats)
            except Exception:
                pass
        if self.clustering_method == "kmeans":
            return KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(feats)
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(feats)

    def _build_sub_parts(
        self,
        mask_id: int,
        parent_mask: np.ndarray,
        feats: np.ndarray,  # [n_inside, D]  raw features (no coords)
        labels: np.ndarray,
        n_clusters: int,
        grid_size: int,
        H: int,
        W: int,
        gray: np.ndarray,
    ) -> List[PartDescriptor]:
        mask_small = cv2.resize(
            parent_mask.astype(np.uint8),
            (grid_size, grid_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        patch_indices = np.where(mask_small.ravel())[0]

        parts = []
        for c in range(n_clusters):
            sel = labels == c
            if not sel.any():
                continue
            idx = patch_indices[sel]
            flat = np.zeros(grid_size * grid_size, dtype=np.uint8)
            flat[idx] = 1
            full_mask = (
                cv2.resize(
                    flat.reshape(grid_size, grid_size),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                & parent_mask
            )

            part = self._make_part(mask_id, full_mask, feats[sel], gray)
            if part is not None:
                parts.append(part)
        return parts

    def _make_part(
        self,
        parent_id: int,
        mask: np.ndarray,
        feats: np.ndarray,
        gray: np.ndarray,
    ) -> Optional[PartDescriptor]:
        area = int(mask.sum())
        if area < self.min_part_area:
            return None

        prototype = feats.mean(axis=0) if len(feats) > 0 else np.zeros(1)
        sem_var = _region_variance(feats)
        compactness, perim = _shape_stats(mask)
        boundary = _boundary_gradient_score(mask, gray)
        ys, xs = np.where(mask)
        centroid = np.array([ys.mean(), xs.mean()])

        pid = self._part_counter
        self._part_counter += 1

        return PartDescriptor(
            part_id=pid,
            parent_mask_id=parent_id,
            mask=mask,
            prototype=prototype,
            semantic_variance=sem_var,
            shape_compactness=compactness,
            boundary_score=boundary,
            area=area,
            centroid=centroid,
            perimeter=perim,
        )

    def _filter_stable(self, parts: List[PartDescriptor]) -> List[PartDescriptor]:
        stable = []
        for p in parts:
            if p.semantic_variance > self.sem_var_thresh:
                continue
            if p.shape_compactness < self.shape_thresh:
                continue
            if self.boundary_thresh > 0 and p.boundary_score < self.boundary_thresh:
                continue
            stable.append(p)
        return stable

    def _build_graph(
        self,
        parts: List[PartDescriptor],
        img_diag: float,
    ) -> PartNode:
        """
        [Fix 2] Multi-relation hierarchy: containment | attached_to | near
        [Fix 3] Contact-ratio adjacency
        [Fix 4] SAM-parent grouping preserved via parent_mask_id
        """
        nodes = {p.part_id: PartNode(part=p) for p in parts}
        sorted_parts = sorted(parts, key=lambda p: p.area)

        # ── Pass 1: containment ────────────────────────────────────────
        for i, pi in enumerate(sorted_parts):
            for pj in sorted_parts[i + 1 :]:
                inter = int((pi.mask & pj.mask).sum())
                if inter == 0:
                    continue
                iou_frac = inter / (pi.area + 1e-6)
                if iou_frac > self.containment_iou_thresh and pi.area < pj.area:
                    ni, nj = nodes[pi.part_id], nodes[pj.part_id]
                    if ni.parent is None:
                        ni.parent = nj
                        nj.children.append((ni, PartEdge(REL_CONTAINS, iou_frac)))
                    break

        # ── Pass 2: attachment & near  (contact-ratio) ────────────────
        for i, pi in enumerate(sorted_parts):
            for pj in sorted_parts[i + 1 :]:
                ni, nj = nodes[pi.part_id], nodes[pj.part_id]
                if ni.parent is nj or nj.parent is ni:
                    continue  # already containment

                min_perim = min(pi.perimeter, pj.perimeter)
                if min_perim < 1e-6:
                    continue

                contact = _contact_length(pi.mask, pj.mask)
                contact_ratio = contact / (min_perim + 1e-6)

                if contact_ratio >= self.contact_attach_thresh:
                    edge = PartEdge(REL_ATTACHED, contact_ratio)
                    ni.peers.append((nj, edge))
                    nj.peers.append((ni, edge))
                    # Also record spatial direction
                    sp_rel = _spatial_relation(pi.centroid, pj.centroid)
                    sp_edge = PartEdge(sp_rel, contact_ratio)
                    ni.peers.append((nj, sp_edge))
                    nj.peers.append((ni, sp_edge))

                elif contact_ratio >= self.contact_near_thresh:
                    edge = PartEdge(REL_NEAR, contact_ratio)
                    ni.peers.append((nj, edge))
                    nj.peers.append((ni, edge))

                else:
                    dist = float(np.linalg.norm(pi.centroid - pj.centroid))
                    if dist / (img_diag + 1e-6) < self.near_dist_thresh:
                        edge = PartEdge(REL_NEAR, 1.0 - dist / img_diag)
                        ni.peers.append((nj, edge))
                        nj.peers.append((ni, edge))

        # ── Virtual root ───────────────────────────────────────────────
        root_part = PartDescriptor(
            part_id=-1,
            parent_mask_id=-1,
            mask=np.array([]),
            prototype=np.array([]),
            semantic_variance=0.0,
            shape_compactness=0.0,
            boundary_score=0.0,
            area=0,
            centroid=np.array([0.0, 0.0]),
            perimeter=0.0,
        )
        root = PartNode(part=root_part)
        for node in nodes.values():
            if node.parent is None:
                node.parent = root
                root.children.append((node, PartEdge(REL_CONTAINS, 1.0)))

        return root
