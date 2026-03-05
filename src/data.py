from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass
class SampleRecord:
    image_path: Path
    label: int


class RetinopathyDataset(Dataset):
    """CSV-backed image dataset for DR grading."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        image_col: str,
        label_col: Optional[str] = None,
        transform=None,
        default_ext: Optional[str] = None,
        label_map: Optional[dict[int, int]] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.default_ext = default_ext
        self.label_map = label_map

        df = pd.read_csv(self.csv_path)
        if image_col not in df.columns:
            raise ValueError(
                f"Column '{image_col}' not found in {self.csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        if label_col and label_col not in df.columns:
            raise ValueError(
                f"Column '{label_col}' not found in {self.csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

        self.samples: list[SampleRecord] = []
        for _, row in df.iterrows():
            image_key = str(row[image_col]).strip()
            image_path = self._resolve_image_path(image_key)
            label = -1 if not label_col else self._map_label(int(row[label_col]))
            self.samples.append(SampleRecord(image_path=image_path, label=label))

    def _map_label(self, raw_label: int) -> int:
        if self.label_map is None:
            return raw_label
        if raw_label not in self.label_map:
            raise ValueError(
                f"Label {raw_label} not found in label_map for dataset {self.csv_path}. "
                f"Available keys: {sorted(self.label_map.keys())}"
            )
        return int(self.label_map[raw_label])

    def _resolve_image_path(self, image_key: str) -> Path:
        candidate = self.image_dir / image_key
        if candidate.exists():
            return candidate

        key_path = Path(image_key)
        if key_path.suffix:
            raise FileNotFoundError(f"Image not found: {candidate}")

        candidates = []
        if self.default_ext:
            ext = self.default_ext if self.default_ext.startswith(".") else f".{self.default_ext}"
            candidates.append(self.image_dir / f"{image_key}{ext}")
        candidates.extend(
            [
                self.image_dir / f"{image_key}.png",
                self.image_dir / f"{image_key}.jpg",
                self.image_dir / f"{image_key}.jpeg",
                self.image_dir / f"{image_key}.tif",
                self.image_dir / f"{image_key}.tiff",
            ]
        )

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not resolve image for key '{image_key}' in {self.image_dir}."
        )

    @property
    def has_labels(self) -> bool:
        return self.label_col is not None

    @property
    def labels(self) -> list[int]:
        return [s.label for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        image = Image.open(rec.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, rec.label


def stratified_split_indices(
    labels: list[int],
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be between 0 and 1. Got {val_split}")

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=np.asarray(labels),
    )
    return train_idx, val_idx
