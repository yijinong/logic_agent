import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import pipeline

from logic_agent.model.segment import MODEL_CFG, SAM2_CKPT

GROUNDING_DINO_PATH = "/data/LLM_Weights/grounding-dino-base"


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(
            image_cv2,
            f"{label}: {score:.2f}",
            (box.xmin, box.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color.tolist(),
            2,
        )

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def plot_detections(
    image: Union[Image.Image, np.ndarray], detections: List[DetectionResult], save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis("off")
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
    plt.show()


def save_segmented_objects(
    image: Union[Image.Image, np.ndarray], detections: List[DetectionResult], output_dir: Union[str, Path]
) -> None:
    image_np = (
        np.array(image.convert("RGB"), copy=True) if isinstance(image, Image.Image) else np.array(image, copy=True)
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label_counts: Dict[str, int] = {}

    for detection in detections:
        if detection.mask is None:
            continue

        label_key = "_".join(detection.label.strip().split()) or "object"
        label_counts[label_key] = label_counts.get(label_key, 0) + 1
        suffix = "" if label_counts[label_key] == 1 else f"_{label_counts[label_key]:03d}"

        x0, y0, x1, y1 = [int(value) for value in detection.box.xyxy]
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_np.shape[1] - 1)
        y1 = min(y1, image_np.shape[0] - 1)

        object_mask = detection.mask.astype(bool)
        masked_image = np.zeros_like(image_np, dtype=np.uint8)
        masked_image[object_mask] = image_np[object_mask]
        cropped_image = masked_image[y0 : y1 + 1, x0 : x1 + 1]

        Image.fromarray(cropped_image).save(output_path / f"{label_key}{suffix}.png")


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))


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


def compute_iou(boxA, boxB):
    xA = max(boxA.xmin, boxB.xmin)
    yA = max(boxA.ymin, boxB.ymin)
    xB = min(boxA.xmax, boxB.xmax)
    yB = min(boxA.ymax, boxB.ymax)

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA.xmax - boxA.xmin) * (boxA.ymax - boxA.ymin)
    areaB = (boxB.xmax - boxB.xmin) * (boxB.ymax - boxB.ymin)

    union = areaA + areaB - inter_area + 1e-6

    return inter_area / union


def remove_duplicate_detections(detections, iou_threshold=0.6):
    detections = sorted(detections, key=lambda x: x.score, reverse=True)
    filtered = []

    for det in detections:
        keep = True

        for kept in filtered:
            iou = compute_iou(det.box, kept.box)

            if iou > iou_threshold:
                keep = False
                break

        if keep:
            filtered.append(det)

    return filtered


def load_image(image_str: str) -> Image.Image:
    image = Image.open(image_str).convert("RGB").resize((1024, 1024))

    return image


def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


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


def detect(
    image: Image.Image, labels: List[str], thresholds: Dict[str, float] = None, detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(
        model=detector_id,
        task="zero-shot-object-detection",
        device=device,
        use_fast=True,
    )

    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=0.1)
    # results = [DetectionResult.from_dict(result) for result in results]

    detections = []
    for r in results:
        label = r["label"].replace(".", "")

        if r["score"] >= thresholds.get(label, 0.3):
            detections.append(DetectionResult.from_dict(r))

    return detections


def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    if not detection_results:
        return detection_results

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ckpt = segmenter_id if segmenter_id is not None else SAM2_CKPT

    sam2_model = build_sam2(MODEL_CFG, model_ckpt, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2_model)

    image_np = (
        np.array(image.convert("RGB"), copy=True) if isinstance(image, Image.Image) else np.array(image, copy=True)
    )
    image_np = np.ascontiguousarray(image_np)
    predictor.set_image(image_np)

    height, width = image_np.shape[:2]

    for detection_result in detection_results:
        box = np.array(detection_result.box.xyxy, dtype=np.float32)
        box[0::2] = np.clip(box[0::2], 0, width - 1)
        box[1::2] = np.clip(box[1::2], 0, height - 1)

        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        best_mask = masks[int(np.argmax(scores))].astype(np.uint8)

        # Keep segmentation strictly inside the corresponding detection box.
        x0, y0, x1, y1 = box.astype(np.int32)
        box_mask = np.zeros_like(best_mask, dtype=np.uint8)
        box_mask[y0 : y1 + 1, x0 : x1 + 1] = 1
        best_mask = (best_mask * box_mask).astype(np.uint8)

        if polygon_refinement:
            polygon = mask_to_polygon(best_mask)
            best_mask = polygon_to_mask(polygon, best_mask.shape)

        detection_result.mask = best_mask

    return detection_results


def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    thresholds: Dict[str, float] = None,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None,
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, thresholds, detector_id)
    detections = remove_duplicate_detections(detections, iou_threshold=0.6)

    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections


def main():
    image = "/home/yijin/projects/logic_agent/data/breakfast_box/train/good/005.png"

    labels = ["banana fruit", "tangerine fruit", "nectarine fruit", "almond chips", "cereal", "breakfast box"]
    thresholds = {
        "breakfast box": 0.5,
        "banana fruit": 0.45,
        "nectarine fruit": 0.45,
        "tangerine fruit": 0.45,
        "cereal": 0.4,
        "almond chips": 0.25,
    }
    threshold = 0.3

    detector_id = GROUNDING_DINO_PATH

    segmentor_id = SAM2_CKPT

    image_array, detections = grounded_segmentation(
        image=image,
        labels=labels,
        thresholds=thresholds,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmentor_id,
    )

    plot_detections(image_array, detections, "test.png")
    save_segmented_objects(image_array, detections, "segmented_objects")


if __name__ == "__main__":
    main()
