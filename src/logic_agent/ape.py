from typing import Any, Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure

from logic_agent.model.segment import MODEL_CFG, SAM2_CKPT, SAM2Segmentor

HF_SAM2_MODEL = "/data/LLM_Weights/sam2.1-hiera-large"
HF_SAM2_FALLBACK = "facebook/sam2.1-hiera-large"


def mask_to_polygon(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a binary mask to the largest contour polygon."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    return [(int(pt[0]), int(pt[1])) for pt in largest_contour.reshape(-1, 2).tolist()]


def polygon_to_mask(polygon: Sequence[Sequence[int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """Convert polygon vertices to a binary mask for the given (H, W)."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    if not polygon:
        return mask

    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask


def refine_masks(masks: List[Dict[str, Any]], polygon_refinement: bool = False) -> List[Dict[str, Any]]:
    """Refine SAM outputs in current pipeline format.

    Expected input: List[Dict], each dict contains a `segmentation` field.
    Returns the same structure with cleaned boolean masks.
    """
    refined: List[Dict[str, Any]] = []
    for m in masks:
        if not isinstance(m, dict):
            continue
        seg = m.get("segmentation")
        if seg is None:
            continue

        if isinstance(seg, torch.Tensor):
            mask_np = seg.detach().cpu().numpy()
        else:
            mask_np = np.asarray(seg)

        if mask_np.ndim > 2:
            mask_np = np.squeeze(mask_np)
        mask_np = (mask_np > 0).astype(np.uint8)

        if polygon_refinement:
            polygon = mask_to_polygon(mask_np)
            mask_np = (polygon_to_mask(polygon, mask_np.shape) > 0).astype(np.uint8)

        out = dict(m)
        out["segmentation"] = mask_np.astype(bool)
        out["area"] = int(mask_np.sum())
        refined.append(out)

    return refined


def show_segmentation_results(image: np.ndarray, masks: List[Dict[str, Any]], borders: bool = False) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for item in masks:
        mask = item.get("segmentation") if isinstance(item, dict) else None
        if mask is None:
            continue
        mask_np = mask.astype(np.uint8)
        if borders:
            contours = measure.find_contours(mask_np, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        else:
            plt.imshow(mask_np, alpha=0.5)
    plt.axis("off")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "/home/yijin/projects/logic_agent/data/splicing_connectors/train/good/000.png"

    segmentator = SAM2Segmentor(model_cfg=MODEL_CFG, model_ckpt=SAM2_CKPT, device=device)
    # stabilizer = MaskStabilizer(min_area_ratio=0.002, max_elongation=15.0)

    img = plt.imread(image_path)
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    sam_res = segmentator.segment(img)
    # masks = stabilizer.process(masks, img.shape[0], img.shape[1])

    print(f"Segmented {len(sam_res)} masks")
    clean_masks = refine_masks(sam_res, True)
    print(f"Refined to {len(clean_masks)} masks after polygon refinement")

    # image_np = np.asarray(image)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    show_segmentation_results(img, clean_masks, borders=True)
    plt.axis("off")
    plt.savefig("sam2_segmentor_test.png")
    plt.show()


if __name__ == "__main__":
    main()
