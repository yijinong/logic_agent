from pathlib import Path
from typing import Literal

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from logic_agent.logging import get_logger

LOGGER = get_logger(__name__)

type Subset = Literal["train", "validation", "test"]


class MVTecLOCODataset(Dataset):
    def __init__(
        self,
        root_dir: Path | str,
        category: str,
        subset: Subset = "train",
        anomaly_type=None,
        transform: torch.nn.Module | None = None,
        load_mask: bool = False,
        binarize_mask: bool = True,
        img_size: int = 256,
    ):
        """
        Construct a MVTec-LOCO dataset to be used in PyTorch dataloader

        :param root_dir: Root directory containing the dataset files
        :type root_dir: Path | str
        :param category: Type of object category
        :type category: str
        :param subset: Subset of the dataset to use, one of "train", "val" or "test"
        :type subset: Subset
        :param anomaly_type: None (all), 'good', 'structural', 'logical`
        :type anomaly_type: str | None
        :param transform: Optional transform to apply to the images, if non provided will default to ToTensor
        :type transform: One of torchvision.transforms.v2, optional
        :param load_mask: If True, load and return GT masks when available
        :type load_mask: bool
        :param binarize_mask: If True, convert GT masks to binary (mask > 0)
        :type binarize_mask: bool
        :param img_size: Size of the image to be resized
        :type img_size: int
        """
        super().__init__()
        self._root = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self._category = category
        self._subset = subset
        if self._subset not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid subset: {self._subset}. Must be one of 'train', 'validation', or 'test'")
        self._anomaly_type = anomaly_type
        self.load_mask = load_mask
        self.binarize_mask = binarize_mask
        self.img_size = img_size

        if transform is None:
            LOGGER.warning("No transform provided, using default [ToImage, ToDtype] transform.")
        self._transform = transform or v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.samples = []
        self.labels = []
        self.defect_types = []
        self.mask_paths = []

        self._load_data()

    def _get_mask_path(self, img_path: Path, defect_type: str) -> Path | None:
        """Resolve GT mask path for a test image according to MVTec-LOCO layout."""
        if defect_type == "good":
            return None

        # MVTec-LOCO mask layout: <category>/ground_truth/<defect_type>/<stem>/000.png
        mask_path = self._root / self._category / "ground_truth" / defect_type / img_path.stem / "000.png"
        return mask_path if mask_path.exists() else None

    def _load_data(self):
        """Load dataset according to the category"""
        category_path = self._root / self._category / self._subset

        if not category_path.exists():
            raise FileNotFoundError(f"Category path {category_path} does not exist.")

        if self._subset in ["train", "validation"]:
            # Training and validation data only contains normal samples
            good_dir = category_path / "good"
            if not good_dir.exists():
                raise FileNotFoundError(f"Good directory {good_dir} does not exist.")

            img_files = sorted(good_dir.glob("*.png"))

            for img in img_files:
                self.samples.append(str(img))
                self.labels.append(0)
                self.defect_types.append("good")
                self.mask_paths.append(None)

        elif self._subset == "test":
            anomaly_types = ["good", "logical_anomalies", "structural_anomalies"]
            if self._anomaly_type:
                if self._anomaly_type not in anomaly_types:
                    raise ValueError(f"Invalid anomaly type for test subset: {self._anomaly_type}")
                anomaly_types = [self._anomaly_type]

            for defect_type in anomaly_types:
                defect_dir = category_path / defect_type
                if not defect_dir.exists():
                    LOGGER.warning(f"Defect directory {defect_dir} does not exist. Skipping.")
                    continue

                img_files = sorted(defect_dir.glob("*.png"))

                for img in img_files:
                    self.samples.append(str(img))
                    self.labels.append(0 if defect_type == "good" else 1)
                    self.defect_types.append(defect_type)
                    self.mask_paths.append(self._get_mask_path(img, defect_type))

            total_samples = len(self.samples)
            if total_samples > 0:
                LOGGER.info(f"Test dataset statistics for {self._category}:")
                LOGGER.info(f"Total samples: {total_samples}")

        else:
            raise ValueError(f"Invalid subset: {self._subset}. Must be train/validation/test")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._transform:
            image = self._transform(image)

        label = self.labels[idx]
        defect_type = self.defect_types[idx]
        mask_path = self.mask_paths[idx]

        result = {
            "image": image,
            "label": label,
            "defect_type": defect_type,
            "img_path": img_path,
            "mask_path": str(mask_path) if mask_path is not None else None,
        }

        if self.load_mask:
            if mask_path is None:
                mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            else:
                mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_np is None:
                    raise FileNotFoundError(f"Failed to read mask at {mask_path}")
                mask_np = cv2.resize(mask_np, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

                if self.binarize_mask:
                    mask_np = (mask_np > 0).astype("float32")
                else:
                    mask_np = mask_np.astype("float32")

                mask = torch.from_numpy(mask_np).unsqueeze(0)

            result["mask"] = mask

        return result

    def get_image_path(self, idx: int) -> Path:
        """Return the image file path for the given sample index as a Path."""
        return Path(self.samples[idx])

    def get_subset_by_defect_type(self, defect_type: str):
        """Get indices for samples of a specific defect type"""
        indices = [i for i, d in enumerate(self.defect_types) if d == defect_type]

        return torch.utils.data.Subset(self, indices)


def create_mvtec_loco_datasets(root_dir: Path | str, category: str, img_size: int = 256, val_split: float = 0.1):
    """
    Create train, validation and test datasets for MVTec-LOCO
    """
    root = Path(root_dir) if isinstance(root_dir, str) else root_dir

    # check if validation folder exists
    val_path = root / category / "validation"
    if val_path.exists():
        train_dataset = MVTecLOCODataset(root, category, subset="train", img_size=img_size)
        val_dataset = MVTecLOCODataset(root, category, subset="validation", img_size=img_size)
    else:
        # Only train and test available, split train for validation
        train_val_dataset = MVTecLOCODataset(root, category, subset="train", img_size=img_size)
        train_size = int((1 - val_split) * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    test_dataset = MVTecLOCODataset(root, category, subset="test", img_size=img_size)

    return train_dataset, val_dataset, test_dataset


def main():
    dataset = MVTecLOCODataset(
        "/home/yijin/projects/logic-agent/data/mvtec_loco_ad", "juice_bottle", subset="test", img_size=256
    )

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    train_ds, val_ds, test_ds = create_mvtec_loco_datasets(
        "/home/yijin/projects/logic-agent/data/mvtec_loco_ad", "juice_bottle", img_size=256, val_split=0.1
    )
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    for i in range(5):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Defect type: {sample['defect_type']}")
        print(f"Image path: {sample['img_path']}")


if __name__ == "__main__":
    main()
