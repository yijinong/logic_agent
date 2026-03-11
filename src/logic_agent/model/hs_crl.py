"""
hs_crl.py  –  Hierarchical Segmentation with Compositional Rule Learning
=========================================================================

Architecture (three levels)
-----------------------------
Scene
 ├─ Object 0  (all parts from SAM mask 0)
 │    ├─ Part 0-0
 │    └─ Part 0-1
 ├─ Object 1  (all parts from SAM mask 1)
 │    └─ Part 1-0
 └─ ...

Objects are formed by grouping PartDescriptors that share the same
parent_mask_id.  The object feature is the mean of its part prototypes.
Objects are then assigned to one of K_obj semantic object types via KMeans,
exactly as parts are assigned to K_part part types.

Six energy terms
----------------
  E_patch       – top-k% MAE of patch features vs nearest part centroid
  E_object      – normalised distance of each part to its centroid
  E_composition – P(child_part_type, relation | parent_part_type)
                  weighted missing-part penalty
  E_graph       – distance of scene-graph embedding to nearest normal prototype
  E_count       – Gaussian z-score of per-object-type count vs training mean/std
  E_spatial     – Mahalanobis distance of per-object-type centroid positions
                  vs training covariance

Total energy
------------
  E = α·E_patch + β·E_object + γ·E_composition
      + δ·E_graph + η·E_count + λ·E_spatial

Default weights emphasise logical structure (count + composition):
  α=0.10, β=0.15, γ=0.30, δ=0.15, η=0.20, λ=0.10

Why count rules matter for splicing_connectors
-----------------------------------------------
Defects like missing_connector, missing_cable, extra_cable, wrong_connector_type
all change the number of connector or cable objects.  E_count directly measures
this deviation from the Gaussian count distribution learned on normal images.

Why spatial rules matter
------------------------
Defects like wrong_cable_location, flipped_connector change where objects appear.
E_spatial measures how far each object centroid is from its learned spatial
distribution N(μ_k, Σ_k) per object type k.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans

from logic_agent.logging import get_logger
from logic_agent.model.hpd import (
    REL_ABOVE,
    REL_ATTACHED,
    REL_BELOW,
    REL_CONTAINS,
    REL_LEFT_OF,
    REL_NEAR,
    REL_RIGHT_OF,
    PartDescriptor,
    PartNode,
)

LOGGER = get_logger(__name__)

ALL_RELATIONS = [REL_CONTAINS, REL_ATTACHED, REL_NEAR, REL_ABOVE, REL_BELOW, REL_LEFT_OF, REL_RIGHT_OF]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ObjectInstance:
    """
    One object = a group of parts that share the same SAM parent_mask_id.

    The object-level feature is the mean of its constituent part prototypes.
    """

    object_id: int  # unique id within this image
    parent_mask_id: int  # SAM mask index
    parts: List["PartInstance"]
    feature: np.ndarray  # mean part prototype [D]
    centroid: np.ndarray  # mean centroid (y, x) in pixel coords, normalised [0,1]
    type_id: int = -1  # assigned object cluster type
    count_energy: float = 0.0  # z-score contribution
    spatial_energy: float = 0.0  # Mahalanobis contribution


@dataclass
class PartInstance:
    """A single part from one image, with its cluster-type assignment."""

    descriptor: PartDescriptor
    type_id: int  # index into the K cluster centroids
    embedding: np.ndarray  # prototype vector [D]
    object_energy: float = 0.0  # distance to cluster centroid


@dataclass
class SceneGraph:
    """Full scene representation for one image."""

    parts: List[PartInstance]
    objects: List[ObjectInstance]
    # (parent_part_type, child_part_type, relation) → count
    relation_counts: Dict[Tuple[int, int, str], int] = field(default_factory=dict)
    patch_energy: float = 0.0
    object_energy: float = 0.0
    composition_energy: float = 0.0
    graph_energy: float = 0.0
    count_energy: float = 0.0
    spatial_energy: float = 0.0
    total_energy: float = 0.0


# ---------------------------------------------------------------------------
# Prototype-based anomaly mapper
# ---------------------------------------------------------------------------


class PrototypeAnomalyMapper:
    """
    E_patch = mean of the top-k% highest per-patch L1 errors against the
    nearest part-type centroid.  Top-k prevents normal patches from diluting
    anomalous regions.
    """

    def __init__(self, top_k_fraction: float = 0.1) -> None:
        self.top_k_fraction = top_k_fraction
        self._centroids: Optional[np.ndarray] = None  # [K, D]

    def set_centroids(self, centroids: np.ndarray) -> None:
        self._centroids = centroids.astype(np.float32)

    # ------------------------------------------------------------------
    def error_map(
        self,
        dense_feats: np.ndarray,  # [N, D]
        grid_size: int,
    ) -> np.ndarray:
        """Compute (grid_size, grid_size) MAE heatmap against nearest centroid."""
        if self._centroids is None:
            return np.zeros((grid_size, grid_size), dtype=np.float32)

        feats = dense_feats.astype(np.float32)
        chunk = 512
        errors = np.empty(len(feats), dtype=np.float32)

        for s in range(0, len(feats), chunk):
            e = min(s + chunk, len(feats))
            batch = feats[s:e]  # [B, D]
            diff = batch[:, None, :] - self._centroids[None, :, :]  # [B, K, D]
            l2 = np.linalg.norm(diff, axis=-1)  # [B, K]
            best = np.argmin(l2, axis=-1)  # [B]
            errors[s:e] = np.abs(batch - self._centroids[best]).mean(axis=-1)

        return errors.reshape(grid_size, grid_size)

    # ------------------------------------------------------------------
    def score(self, dense_feats: np.ndarray) -> float:
        """[Fix 5] Top-K mean patch error."""
        if self._centroids is None or len(dense_feats) == 0:
            return 0.0
        N = len(dense_feats)
        grid_size = int(math.isqrt(N))
        flat = self.error_map(dense_feats, grid_size).ravel()
        k = max(1, int(len(flat) * self.top_k_fraction))
        return float(np.partition(flat, -k)[-k:].mean())


# ---------------------------------------------------------------------------
# Compositional Rule Learner (spatial + importance-weighted)
# ---------------------------------------------------------------------------


class CompositionalRuleLearner:
    """
    learns p(child_part_type, relation | par

    Rule key: (parent_type, child_type, relation_str)
    Missing expected children are penalised proportionally to their learned
    probability (importance-weighted penalty).
    """

    def __init__(
        self,
        n_part_types: int,
        presence_threshold: float = 0.5,
        laplace_alpha: float = 0.5,
        eps: float = 1e-3,
    ):
        self.K = n_part_types
        self.presence_threshold = presence_threshold
        self.laplace_alpha = laplace_alpha
        self.eps = eps

        # (parent, child, relation) → co-occurrence count
        self._co_occur: Dict[Tuple[int, int, str], float] = defaultdict(float)
        # parent → count of images where this parent type appeared
        self._parent_count: Dict[int, float] = defaultdict(float)

    # ------------------------------------------------------------------
    def update(self, sg: SceneGraph) -> None:
        """Register one normal image's scene graph."""
        seen_parents: set = set()
        for (pt, ct, rel), cnt in sg.relation_counts.items():
            if cnt > 0:
                self._co_occur[(pt, ct, rel)] += 1.0
                seen_parents.add(pt)
        for pt in seen_parents:
            self._parent_count[pt] += 1.0

    def conditional_prob(self, parent_type: int, child_type: int, relation: str) -> float:
        n_combos = self.K * len(ALL_RELATIONS)
        count = self._co_occur.get((parent_type, child_type, relation), 0.0)
        total = self._parent_count.get(parent_type, 0.0)
        return (count + self.laplace_alpha) / (total + self.laplace_alpha * n_combos)

    def expected_rules(self, parent_type: int) -> List[Tuple[int, str, float]]:
        """Return [(child_type, relation, prob)] with prob >= threshold."""
        out = []
        for ct in range(self.K):
            for rel in ALL_RELATIONS:
                p = self.conditional_prob(parent_type, ct, rel)
                if p >= self.presence_threshold:
                    out.append((ct, rel, p))
        return out

    def composition_energy(
        self,
        parent_type: int,
        observed_child_rels: List[Tuple[int, str]],
    ) -> float:
        """
        [Fix 6] Importance-weighted missing-part penalty.

        E = -1/|expected| * Σ_j  importance_j * log P(child_j, rel_j | parent)

        where importance_j = P(child_j, rel_j | parent)  for expected rules,
        so high-confidence expected relations contribute a stronger penalty
        when absent.
        """
        expected = self.expected_rules(parent_type)
        if not expected:
            return 0.0

        observed_set = set(observed_child_rels)
        total_weight = 0.0
        weighted_logp = 0.0

        for ct, rel, p in expected:
            importance = p  # [Fix 6] weight by rule probability
            total_weight += importance
            if (ct, rel) in observed_set:
                weighted_logp += importance * math.log(max(p, self.eps))
            else:
                weighted_logp += importance * math.log(self.eps)

        if total_weight < 1e-9:
            return 0.0
        return float(-weighted_logp / total_weight)

    # ------------------------------------------------------------------
    def scene_composition_energy(self, sg: SceneGraph) -> float:
        """Mean composition energy over all parent types in the scene graph."""
        # Collect observed (child, rel) per parent from relation_counts
        parent_to_obs: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for (pt, ct, rel), cnt in sg.relation_counts.items():
            if cnt > 0:
                parent_to_obs[pt].append((ct, rel))

        if not parent_to_obs:
            return 0.0

        total, n = 0.0, 0
        for pt, obs in parent_to_obs.items():
            total += self.composition_energy(pt, obs)
            n += 1
        return total / n


# ---------------------------------------------------------------------------
# E_graph – graph prototype memory
# ---------------------------------------------------------------------------


class GraphPrototypeMemory:
    """
    Embeds each scene graph as a fixed-length histogram vector and scores a
    query by its L2 distance to the nearest stored normal embedding.

    Embedding = concat(
        part_type_histogram      [K_part]
        relation_type_histogram  [|ALL_RELATIONS|]
        spatial_histogram        [4]   (above/below/left_of/right_of fractions)
    )
    """

    def __init__(self, n_part_types: int):
        self.K = n_part_types
        self._prototypes: List[np.ndarray] = []
        self._std: float = 1.0

    def _embed(self, sg: SceneGraph) -> np.ndarray:
        part_hist = np.zeros(self.K, dtype=np.float32)
        rel_hist = np.zeros(len(ALL_RELATIONS), dtype=np.float32)
        spatial_hist = np.zeros(4, dtype=np.float32)
        sp_map = {REL_ABOVE: 0, REL_BELOW: 1, REL_LEFT_OF: 2, REL_RIGHT_OF: 3}

        for pi in sg.parts:
            if pi.type_id < self.K:
                part_hist[pi.type_id] += 1

        for (_, _, rel), cnt in sg.relation_counts.items():
            if rel in ALL_RELATIONS:
                rel_hist[ALL_RELATIONS.index(rel)] += cnt
            if rel in sp_map:
                spatial_hist[sp_map[rel]] += cnt

        def _norm(v):
            s = v.sum()
            return v / (s + 1e-6)

        return np.concatenate([_norm(part_hist), _norm(rel_hist), _norm(spatial_hist)])

    def update(self, sg: SceneGraph) -> None:
        self._prototypes.append(self._embed(sg))

    def finalise(self) -> None:
        """Compute std of pairwise distances for normalisation."""
        if len(self._prototypes) < 2:
            self._std = 1.0
            return
        P = np.stack(self._prototypes)
        D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
        mask = np.triu(np.ones_like(D, dtype=bool), k=1)
        self._std = float(D[mask].std()) + 1e-6

    def score(self, sg: SceneGraph) -> float:
        if not self._prototypes:
            return 0.0
        q = self._embed(sg)
        dists = np.linalg.norm(np.stack(self._prototypes) - q[None], axis=1)
        return float(np.min(dists) / self._std)


# ---------------------------------------------------------------------------
# Part-type assigner
# ---------------------------------------------------------------------------


class PartTypeAssigner:
    """
    Assigns each PartDescriptor to one of K semantic part types by
    nearest-centroid matching in embedding space.
    """

    def __init__(self, n_part_types: int = 16):
        self.K = n_part_types
        self._centroids: Optional[np.ndarray] = None  # [K, D]
        self._centroid_stds: Optional[np.ndarray] = None  # [K]

    @staticmethod
    def _to_1d_proto(proto: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convert prototype tensor/array to a contiguous float32 1D vector."""
        if isinstance(proto, torch.Tensor):
            arr = proto.detach().cpu().numpy()
        else:
            arr = np.asarray(proto)
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------------
    def fit(self, all_parts: List[PartDescriptor]) -> None:
        """Cluster all reference parts to obtain K centroids."""
        if not all_parts:
            raise ValueError("all_parts is empty")

        proto_vecs = [self._to_1d_proto(p.prototype) for p in all_parts]
        dims = [int(v.shape[0]) for v in proto_vecs]
        if not dims:
            raise ValueError("No prototype vectors found")

        expected_dim = max(set(dims), key=dims.count)
        valid = [v for v in proto_vecs if v.shape[0] == expected_dim]
        dropped = len(proto_vecs) - len(valid)
        if dropped > 0:
            LOGGER.warning(
                f"Dropping {dropped} parts with inconsistent prototype dims. "
                f"Expected {expected_dim}, saw {sorted(set(dims))}"
            )

        if not valid:
            raise ValueError(
                "No valid part prototypes available after dimension filtering. "
                "Check HPD output and segmentation quality."
            )

        protos = np.stack(valid, axis=0)  # [M, D]
        n_clusters = min(self.K, len(valid))

        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = km.fit_predict(protos)

        self._centroids = km.cluster_centers_  # [K, D]
        # Per-cluster std of distances (for calibration)
        stds = []
        for k in range(n_clusters):
            members = protos[labels == k]
            if len(members) > 1:
                dists = np.linalg.norm(members - self._centroids[k], axis=1)
                stds.append(float(dists.std()))
            else:
                stds.append(1.0)
        self._centroid_stds = np.array(stds, dtype=np.float32)

        LOGGER.info(f"PartTypeAssigner fitted with {n_clusters} clusters from {len(all_parts)} parts")

    # ------------------------------------------------------------------
    def assign(self, part: PartDescriptor) -> Tuple[int, float]:
        """
        Returns (type_id, object_energy).

        object_energy = normalised distance to nearest centroid.
        """
        if self._centroids is None or self._centroid_stds is None:
            raise RuntimeError("Call fit() before assign().")

        proto = self._to_1d_proto(part.prototype)
        if proto.shape[0] != self._centroids.shape[1]:
            raise ValueError(
                f"Prototype dim mismatch at assign(): got {proto.shape[0]}, expected {self._centroids.shape[1]}"
            )

        dists = np.linalg.norm(self._centroids - proto[None], axis=1)
        type_id = int(np.argmin(dists))
        energy = float(dists[type_id] / (self._centroid_stds[type_id] + 1e-8))
        return type_id, energy

    # ------------------------------------------------------------------
    def assign_all(self, parts: List[PartDescriptor]) -> List[PartInstance]:
        out: List[PartInstance] = []
        skipped = 0
        for p in parts:
            try:
                t, e = self.assign(p)
            except ValueError:
                skipped += 1
                continue

            out.append(
                PartInstance(
                    descriptor=p,
                    type_id=t,
                    embedding=p.prototype,
                    object_energy=e,
                )
            )

        if skipped > 0:
            LOGGER.warning(f"Skipped {skipped} parts with incompatible prototype dimensions during assignment")
        return out


# ---------------------------------------------------------------------------
# Object-type assigner
# ---------------------------------------------------------------------------


class ObjectTypeAssigner:
    """
    Groups PartInstances by SAM parent_mask_id into ObjectInstances, then
    assigns each object to one of K_obj object-type clusters.

    The object feature = mean of its part prototype embeddings.
    The object centroid = normalised mean of its part centroids (in [0,1]²).
    """

    def __init__(self, n_object_types: int = 8):
        self.K = n_object_types
        self._centroids: Optional[np.ndarray] = None  # [K, D]
        self._centroid_stds: Optional[np.ndarray] = None  # [K]

    # ------------------------------------------------------------------
    @staticmethod
    def group_parts_into_objects(
        part_instances: List[PartInstance],
        img_h: int,
        img_w: int,
    ) -> List[ObjectInstance]:
        """
        Collect all parts that share the same parent_mask_id into one
        ObjectInstance.  Centroid is normalised to [0,1]² so spatial
        rules are image-size invariant.
        """
        groups: Dict[int, List[PartInstance]] = defaultdict(list)
        for pi in part_instances:
            groups[pi.descriptor.parent_mask_id].append(pi)

        objects = []
        for obj_id, (mask_id, members) in enumerate(sorted(groups.items())):
            protos = np.stack([pi.embedding for pi in members])  # [m, D]
            feat = protos.mean(axis=0)  # [D]
            centroids_px = np.stack([pi.descriptor.centroid for pi in members])  # [m, 2]
            centroid_norm = centroids_px.mean(axis=0) / np.array([img_h + 1e-6, img_w + 1e-6], dtype=np.float32)
            objects.append(
                ObjectInstance(
                    object_id=obj_id,
                    parent_mask_id=mask_id,
                    parts=members,
                    feature=feat.astype(np.float32),
                    centroid=centroid_norm.astype(np.float32),
                )
            )
        return objects

    # ------------------------------------------------------------------
    def fit(self, all_objects: List[ObjectInstance]) -> None:
        if not all_objects:
            return
        feats = np.stack([o.feature for o in all_objects])
        n_clusters = min(self.K, len(all_objects))
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = km.fit_predict(feats)
        self._centroids = km.cluster_centers_.astype(np.float32)
        stds = []
        for k in range(n_clusters):
            m = feats[labels == k]
            stds.append(float(np.linalg.norm(m - self._centroids[k], axis=1).std()) if len(m) > 1 else 1.0)
        self._centroid_stds = np.array(stds, dtype=np.float32)
        LOGGER.info(f"ObjectTypeAssigner: {n_clusters} clusters from {len(all_objects)} objects")

    # ------------------------------------------------------------------
    def assign(self, obj: ObjectInstance) -> Tuple[int, float]:
        if self._centroids is None or self._centroid_stds is None:
            raise RuntimeError("Call fit() before assign().")
        dists = np.linalg.norm(self._centroids - obj.feature[None], axis=1)
        type_id = int(np.argmin(dists))
        energy = float(dists[type_id] / (self._centroid_stds[type_id] + 1e-8))
        return type_id, energy

    def assign_all(self, objects: List[ObjectInstance]) -> None:
        """Mutates each ObjectInstance in-place."""
        for obj in objects:
            obj.type_id, _ = self.assign(obj)


# ---------------------------------------------------------------------------
# E_count – object count rule learner
# ---------------------------------------------------------------------------


class CountRuleLearner:
    """
    Learns the per-object-type count distribution N(μ_k, σ_k) on normal images.

    At inference:
        E_count = Σ_k  |count_k - μ_k| / (σ_k + ε)

    Directly detects:
        missing_connector, missing_cable, extra_cable,
        wrong_connector_type_*, 0/N_nectarines anomalies, etc.
    """

    def __init__(self, n_object_types: int):
        self.K = n_object_types
        # Accumulate counts per image, per type
        self._counts_per_image: List[np.ndarray] = []  # each [K]
        # Safe defaults allow scoring before finalise() during fit pass 2.
        self._mu = np.zeros(self.K, dtype=np.float32)
        self._sigma = np.ones(self.K, dtype=np.float32)

    def update(self, objects: List[ObjectInstance]) -> None:
        """Register the object-type counts for one normal image."""
        counts = np.zeros(self.K, dtype=np.float32)
        for obj in objects:
            if 0 <= obj.type_id < self.K:
                counts[obj.type_id] += 1
        self._counts_per_image.append(counts)

    def finalise(self) -> None:
        """Compute μ and σ for each object type over all normal images."""
        if not self._counts_per_image:
            self._mu = np.zeros(self.K, dtype=np.float32)
            self._sigma = np.ones(self.K, dtype=np.float32)
            return
        M = np.stack(self._counts_per_image)  # [N_images, K]
        self._mu = M.mean(axis=0).astype(np.float32)
        self._sigma = (M.std(axis=0) + 1e-2).astype(np.float32)
        LOGGER.info("CountRuleLearner: μ=" + str(np.round(self._mu, 2)))

    def score(self, objects: List[ObjectInstance]) -> float:
        """
        Compute E_count for one image.

        Returns the mean normalised absolute deviation across all K types.
        Types with both μ≈0 and count=0 contribute 0 (correctly absent types
        are not penalised).
        """
        counts = np.zeros(self.K, dtype=np.float32)
        for obj in objects:
            if 0 <= obj.type_id < self.K:
                counts[obj.type_id] += 1

        # Only penalise types that normally appear OR appear now
        active = (self._mu > 0.1) | (counts > 0)
        if not active.any():
            return 0.0

        deviations = np.abs(counts - self._mu) / self._sigma
        return float(deviations[active].mean())

    def explain(self, objects: List[ObjectInstance]) -> List[str]:
        counts = np.zeros(self.K, dtype=np.float32)
        for obj in objects:
            if 0 <= obj.type_id < self.K:
                counts[obj.type_id] += 1
        msgs = []
        for k in range(self.K):
            dev = abs(counts[k] - self._mu[k]) / self._sigma[k]
            if dev > 2.0:
                msgs.append(f"obj_type_{k}: count={int(counts[k])} expected≈{self._mu[k]:.1f} (z={dev:.1f}σ)")
        return msgs


# ---------------------------------------------------------------------------
# E_spatial – spatial distribution rule learner
# ---------------------------------------------------------------------------


class SpatialRuleLearner:
    """
    Learns the 2-D centroid distribution p(x,y | object_type) ≈ N(μ_k, Σ_k)
    for each object type k.

    At inference, for each object of type k:
        d_k = (c - μ_k)ᵀ Σ_k⁻¹ (c - μ_k)   (Mahalanobis distance)

    E_spatial = mean over all objects of sqrt(d_k) / normaliser

    Detects:
        wrong_cable_location, compartments_swapped, flipped_connector,
        fruit_outside_bowl, etc.
    """

    def __init__(self, n_object_types: int):
        self.K = n_object_types
        # [type_id] → list of centroid vectors [2]
        self._centroids_by_type: Dict[int, List[np.ndarray]] = defaultdict(list)
        # Safe defaults allow scoring before finalise() during fit pass 2.
        self._mu: Dict[int, np.ndarray] = {}
        self._inv_cov: Dict[int, np.ndarray] = {}
        self._scale: Dict[int, float] = {}

    def update(self, objects: List[ObjectInstance]) -> None:
        for obj in objects:
            if 0 <= obj.type_id < self.K:
                self._centroids_by_type[obj.type_id].append(obj.centroid.copy())

    def finalise(self) -> None:
        """Estimate μ_k and Σ_k for each observed type."""
        self._mu: Dict[int, np.ndarray] = {}
        self._inv_cov: Dict[int, np.ndarray] = {}
        self._scale: Dict[int, float] = {}

        for k, pts in self._centroids_by_type.items():
            pts_arr = np.stack(pts)  # [N, 2]
            mu = pts_arr.mean(axis=0)
            self._mu[k] = mu

            if len(pts_arr) >= 3:
                cov = np.cov(pts_arr.T)
                cov += np.eye(2) * 1e-4  # regularise
                try:
                    inv = np.linalg.inv(cov)
                    # Scale: mean Mahalanobis distance on training set
                    ds = [float(np.sqrt((p - mu) @ inv @ (p - mu))) for p in pts_arr]
                    self._inv_cov[k] = inv
                    self._scale[k] = float(np.mean(ds)) + 1e-6
                except np.linalg.LinAlgError:
                    self._inv_cov[k] = np.eye(2)
                    self._scale[k] = 1.0
            else:
                # Too few samples: use identity / spread as fallback
                spread = float(np.std(pts_arr)) + 1e-3
                self._inv_cov[k] = np.eye(2) / (spread**2)
                self._scale[k] = 1.0

        LOGGER.info(f"SpatialRuleLearner: fitted {len(self._mu)} object types")

    def score(self, objects: List[ObjectInstance]) -> float:
        """Mean normalised Mahalanobis distance over all objects."""
        scores = []
        for obj in objects:
            k = obj.type_id
            if k not in self._mu:
                continue
            d = obj.centroid - self._mu[k]
            mh = float(np.sqrt(d @ self._inv_cov[k] @ d))
            scores.append(mh / self._scale[k])
        return float(np.mean(scores)) if scores else 0.0

    def explain(self, objects: List[ObjectInstance]) -> List[str]:
        msgs = []
        for obj in objects:
            k = obj.type_id
            if k not in self._mu:
                continue
            d = obj.centroid - self._mu[k]
            mh = float(np.sqrt(d @ self._inv_cov[k] @ d)) / self._scale[k]
            if mh > 2.0:
                msgs.append(
                    f"obj_type_{k} centroid at "
                    f"({obj.centroid[0]:.3f}, {obj.centroid[1]:.3f}) "
                    f"deviates from learned position "
                    f"({self._mu[k][0]:.3f}, {self._mu[k][1]:.3f}) "
                    f"(Mahalanobis={mh:.1f})"
                )
        return msgs


# ---------------------------------------------------------------------------
# Scene graph builder  (object-scoped, relation-aware)
# ---------------------------------------------------------------------------


def build_scene_graph(
    part_instances: List[PartInstance],
    objects: List[ObjectInstance],
    hpd_root: PartNode,
) -> SceneGraph:
    """
    Build a SceneGraph from typed parts and the HPD containment tree.

    relation_counts keys: (parent_part_type, child_part_type, relation_str)
    """
    id_to_inst = {pi.descriptor.part_id: pi for pi in part_instances}

    # Group part instances by their SAM parent  [Fix 4]
    obj_to_parts: Dict[int, List[PartInstance]] = defaultdict(list)
    for pi in part_instances:
        obj_to_parts[pi.descriptor.parent_mask_id].append(pi)

    relation_counts: Dict[Tuple[int, int, str], int] = defaultdict(int)

    def _walk(node: PartNode) -> None:
        if node.part.part_id == -1:  # virtual root
            for child_node, _ in node.children:
                _walk(child_node)
            return

        parent_inst = id_to_inst.get(node.part.part_id)
        if parent_inst is None:
            return

        # Hierarchical children
        for child_node, edge in node.children:
            child_inst = id_to_inst.get(child_node.part.part_id)
            if child_inst is not None:
                relation_counts[(parent_inst.type_id, child_inst.type_id, edge.relation)] += 1
                _walk(child_node)

        # Peer relations (attachment, near, spatial)
        for peer_node, edge in node.peers:
            peer_inst = id_to_inst.get(peer_node.part.part_id)
            if peer_inst is not None:
                relation_counts[(parent_inst.type_id, peer_inst.type_id, edge.relation)] += 1

    _walk(hpd_root)

    return SceneGraph(
        parts=part_instances,
        objects=objects,
        relation_counts=dict(relation_counts),
    )


# ---------------------------------------------------------------------------
# Main HS-CRL detector
# ---------------------------------------------------------------------------


class HSCRL:
    """
    Hierarchical Segmentation with Compositional Rule Learning.

    Energy = α·E_patch + β·E_object + γ·E_composition
           + δ·E_graph + η·E_count + λ·E_spatial

    Parameters
    ----------
    n_part_types        : K part clusters
    n_object_types      : K object clusters (for count + spatial rules)
    alpha               : E_patch weight
    beta                : E_object weight
    gamma               : E_composition weight
    delta               : E_graph weight
    eta                 : E_count weight
    lam                 : E_spatial weight
    threshold_k         : threshold = μ + k·σ of normal scores
    presence_threshold  : min P for a part-relation rule to be "expected"
    top_k_fraction      : top-k% patches for E_patch
    """

    def __init__(
        self,
        n_part_types: int = 16,
        n_object_types: int = 8,
        alpha: float = 0.10,
        beta: float = 0.15,
        gamma: float = 0.30,
        delta: float = 0.15,
        eta: float = 0.20,
        lam: float = 0.10,
        threshold_k: float = 2.5,
        presence_threshold: float = 0.5,
        top_k_fraction: float = 0.10,
    ):
        self.n_part_types = n_part_types
        self.n_object_types = n_object_types
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.lam = lam
        self.threshold_k = threshold_k

        self.proto_mapper = PrototypeAnomalyMapper(top_k_fraction=top_k_fraction)
        self.part_assigner = PartTypeAssigner(n_part_types=n_part_types)
        self.obj_assigner = ObjectTypeAssigner(n_object_types=n_object_types)
        self.rule_learner = CompositionalRuleLearner(
            n_part_types=n_part_types,
            presence_threshold=presence_threshold,
        )
        self.graph_memory = GraphPrototypeMemory(n_part_types=n_part_types)
        self.count_learner = CountRuleLearner(n_object_types=n_object_types)
        self.spatial_learner = SpatialRuleLearner(n_object_types=n_object_types)

        self._threshold: Optional[float] = None
        self._normal_scores: List[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, image_results: List[Dict]) -> None:
        """
        Two-pass fitting:

        Pass 1  – fit part-type and object-type assigners on all reference parts
        Pass 2  – build scene graphs, update all rule learners, compute
                  calibration energies
        """
        if not image_results:
            raise ValueError("image_results is empty")

        # ── Pass 1a: fit part-type assigner ───────────────────────────
        all_parts = [p for r in image_results for p in r["parts"]]
        if not all_parts:
            LOGGER.warning("No parts found across reference images.")
            return
        self.part_assigner.fit(all_parts)
        centroids = self.part_assigner._centroids
        if centroids is None:
            raise RuntimeError("PartTypeAssigner fit completed without centroids.")
        self.proto_mapper.set_centroids(centroids)

        # ── Pass 1b: assign part types, form objects, fit object assigner
        H_W_per_image = [r["img_shape"] for r in image_results]

        all_objects_flat: List[ObjectInstance] = []
        image_part_insts: List[List[PartInstance]] = []
        image_objects: List[List[ObjectInstance]] = []

        for r, (H, W) in zip(image_results, H_W_per_image):
            pinsts = self.part_assigner.assign_all(r["parts"])
            objects = ObjectTypeAssigner.group_parts_into_objects(pinsts, H, W)
            image_part_insts.append(pinsts)
            image_objects.append(objects)
            all_objects_flat.extend(objects)

        self.obj_assigner.fit(all_objects_flat)

        # Assign object types now that the assigner is fitted
        for objects in image_objects:
            self.obj_assigner.assign_all(objects)

        # ── Pass 2: update all rule learners ──────────────────────────
        normal_energies: List[float] = []

        for r, pinsts, objects in zip(image_results, image_part_insts, image_objects):
            feats_np = r["dense_feats"].cpu().numpy()
            sg = build_scene_graph(pinsts, objects, r["root"])
            self.rule_learner.update(sg)
            self.graph_memory.update(sg)
            self.count_learner.update(objects)
            self.spatial_learner.update(objects)
            sg = self._score_scene_graph(sg, feats_np)
            normal_energies.append(sg.total_energy)

        # Finalise learners that need a post-pass step
        self.graph_memory.finalise()
        self.count_learner.finalise()
        self.spatial_learner.finalise()

        # ── Re-score with finalised learners for accurate calibration ──
        normal_energies = []
        for r, pinsts, objects in zip(image_results, image_part_insts, image_objects):
            feats_np = r["dense_feats"].cpu().numpy()
            sg = build_scene_graph(pinsts, objects, r["root"])
            sg = self._score_scene_graph(sg, feats_np)
            normal_energies.append(sg.total_energy)

        mu = float(np.mean(normal_energies))
        sig = float(np.std(normal_energies))
        # self._threshold = mu + self.threshold_k * sig
        # FIX: Use percentile-based threshold for better robustness to outliers in normal scores
        # self._threshold = float(np.percentile(normal_energies, 100 * (1 - 1 / (1 + math.exp(-self.threshold_k)))))
        # self._threshold = float(np.percentile(normal_energies, 100 * (1 - 1 / (1 + math.exp(-self.threshold_k)))))
        self._threshold = float(np.percentile(normal_energies, 95))
        self._normal_scores = normal_energies

        LOGGER.info(
            f"HSCRL fitted | {len(image_results)} images | μ={mu:.4f} σ={sig:.4f} threshold={self._threshold:.4f}"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image_result: Dict) -> Dict:
        """
        Score one query image.

        Returns dict with:
            is_anomaly, total_energy,
            patch_energy, object_energy, composition_energy,
            graph_energy, count_energy, spatial_energy,
            threshold, violations, anomaly_map (only if anomalous)
        """
        if self._threshold is None:
            raise RuntimeError("Call fit() before predict().")

        feats_np = image_result["dense_feats"].cpu().numpy()
        H, W = image_result["img_shape"]

        pinsts = self.part_assigner.assign_all(image_result["parts"])
        objects = ObjectTypeAssigner.group_parts_into_objects(pinsts, H, W)
        self.obj_assigner.assign_all(objects)

        sg = build_scene_graph(pinsts, objects, image_result["root"])
        sg = self._score_scene_graph(sg, feats_np)

        is_anomaly = sg.total_energy > self._threshold
        violations = self._explain(sg) if is_anomaly else []

        result = {
            "is_anomaly": is_anomaly,
            "total_energy": sg.total_energy,
            "patch_energy": sg.patch_energy,
            "object_energy": sg.object_energy,
            "composition_energy": sg.composition_energy,
            "graph_energy": sg.graph_energy,
            "count_energy": sg.count_energy,
            "spatial_energy": sg.spatial_energy,
            "threshold": self._threshold,
            "violations": violations,
            "scene_graph": sg,
        }

        if is_anomaly:
            result["anomaly_map"] = self._build_anomaly_map(pinsts, feats_np, H, W)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_scene_graph(self, sg: SceneGraph, feats_np: np.ndarray) -> SceneGraph:
        sg.patch_energy = self.proto_mapper.score(feats_np)
        sg.object_energy = float(np.mean([pi.object_energy for pi in sg.parts])) if sg.parts else 0.0
        sg.composition_energy = self.rule_learner.scene_composition_energy(sg)
        sg.graph_energy = self.graph_memory.score(sg)
        sg.count_energy = self.count_learner.score(sg.objects)
        sg.spatial_energy = self.spatial_learner.score(sg.objects)
        sg.total_energy = (
            self.alpha * sg.patch_energy
            + self.beta * sg.object_energy
            + self.gamma * sg.composition_energy
            + self.delta * sg.graph_energy
            + self.eta * sg.count_energy
            + self.lam * sg.spatial_energy
        )
        return sg

    def _explain(self, sg: SceneGraph) -> List[str]:
        violations: List[str] = []

        # Composition violations
        parent_to_obs: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for (pt, ct, rel), cnt in sg.relation_counts.items():
            if cnt > 0:
                parent_to_obs[pt].append((ct, rel))
        for pt, obs in parent_to_obs.items():
            obs_set = set(obs)
            expected = self.rule_learner.expected_rules(pt)
            for ct, rel, p in expected:
                if (ct, rel) not in obs_set:
                    violations.append(
                        f"COMPOSITION: part_type_{pt} missing expected part_type_{ct} via '{rel}' (P={p:.2f})"
                    )

        # High object-level energy
        for pi in sg.parts:
            if pi.object_energy > 3.0:
                violations.append(
                    f"APPEARANCE: part_type_{pi.type_id} distant from prototype (energy={pi.object_energy:.2f})"
                )

        # Count violations
        violations.extend(f"COUNT: {m}" for m in self.count_learner.explain(sg.objects))

        # Spatial violations
        violations.extend(f"SPATIAL: {m}" for m in self.spatial_learner.explain(sg.objects))

        return violations

    def _build_anomaly_map(
        self,
        part_instances: List[PartInstance],
        feats_np: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        N = feats_np.shape[0]
        grid_size = int(math.isqrt(N))
        feats_grid = feats_np.reshape(grid_size, grid_size, -1)
        amap = np.zeros((H, W), dtype=np.float32)

        for pi in part_instances:
            centroids = self.part_assigner._centroids
            if centroids is None:
                return amap
            centroid = centroids[pi.type_id]
            mask_full = pi.descriptor.mask
            mask_grid = cv2.resize(
                mask_full.astype(np.uint8),
                (grid_size, grid_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if not mask_grid.any():
                continue
            patch_feats = feats_grid[mask_grid]
            mae_per_patch = np.abs(patch_feats - centroid).mean(axis=1)
            err_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
            err_grid[mask_grid] = mae_per_patch
            err_full = cv2.resize(err_grid, (W, H), interpolation=cv2.INTER_LINEAR)
            amap = np.where(mask_full, np.maximum(amap, err_full), amap)

        return amap
