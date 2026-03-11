from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from matplotlib import pyplot as plt
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor  # ← NEW: for point-prompted segmentation
from scipy.ndimage import maximum_filter  # ← NEW: for peak finding
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from logic_agent.dataset.mvtec_loco import MVTecLOCODataset
from logic_agent.logging import get_logger

LOGGER = get_logger(__name__)

SAM2_CKPT = "/data/LLM_Weights/sam2.1-hiera-large/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


@dataclass
class MaskDescriptor:
    mask_id: int
    sam_mask: np.ndarray
    centroid: np.ndarray
    area: float
    prototype: np.ndarray  # Pooled dense feature
    global_cls: torch.Tensor


# ---------------------------------------------------------------------------
# Existing: bottom-up automatic segmentor (unchanged)
# ---------------------------------------------------------------------------


class SAM2Segmentor(nn.Module):
    def __init__(
        self,
        model_cfg: str = MODEL_CFG,
        model_ckpt: str = SAM2_CKPT,
        device: torch.device | str = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.model = self._load_model(model_cfg, model_ckpt, device)

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.95,
            min_mask_region_area=75,
            use_m2m=True,
        )

    def _load_model(self, model_cfg, model_ckpt, device):
        str_device = device if isinstance(device, str) else device.type
        sam2_model = build_sam2(model_cfg, model_ckpt, device=str_device, apply_postprocessing=False)
        sam2_model.eval()
        return sam2_model

    @torch.no_grad()
    def segment(self, img_np: np.ndarray) -> list:
        masks = self.mask_generator.generate(img_np)
        return masks


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


class MaskStabilizer:
    def __init__(self, min_area_ratio: float = 0.0005, max_elongation: float = 15.0):
        self.min_area_ratio = min_area_ratio
        self.max_elongation = max_elongation
        self.kernel = np.ones((3, 3), np.uint8)

    def process(self, sam_masks, H, W):
        stable = []
        for m in sam_masks:
            mask_np = m["segmentation"].astype(np.uint8)
            cleaned = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, self.kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)
            new_area = np.sum(cleaned)
            area_ratio = new_area / (H * W)
            if area_ratio < self.min_area_ratio:
                continue
            elongation = self._compute_elongation(cleaned)
            if elongation > self.max_elongation:
                continue
            m["segmentation"] = cleaned.astype(bool)
            m["area"] = new_area
            stable.append(m)
        return stable

    def _compute_elongation(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return float("inf")
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            return float("inf")
        ellipse = cv2.fitEllipse(cnt)
        _, axes, _ = ellipse
        return max(axes) / (min(axes) + 1e-6)


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0

    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ax.imshow(img)


def main() -> None:
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
        subset="test",
        img_size=1024,
        transform=transform,
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    sam2_model = SAM2Segmentor(MODEL_CFG, SAM2_CKPT, device=device)

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device)
            orig_images = batch["image"].cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(images.shape[0]):
                img_np_float = orig_images[i]
                img_np = (img_np_float * 255).clip(0, 255).astype(np.uint8)
                sam_res = sam2_model.segment(img_np)
                LOGGER.info(f"Image {i}: {len(sam_res)} objects segmented before stabilization.")
                if not sam_res:
                    continue

                clean_sam_res = refine_masks(sam_res, True)

                clean_sam_res = MaskStabilizer(min_area_ratio=0.002, max_elongation=15.0).process(
                    sam_res, img_np.shape[0], img_np.shape[1]
                )
                LOGGER.info(f"Image {i}: {len(clean_sam_res)} objects after stabilization.")
                plt.figure(figsize=(10, 10))
                plt.imshow(img_np)
                show_anns(clean_sam_res, borders=True)
                plt.axis("off")
                plt.savefig(f"sam2_segmentor_{i}.png")
                plt.show()
            break


if __name__ == "__main__":
    main()
