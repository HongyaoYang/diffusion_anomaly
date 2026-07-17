"""
Read NIfTI files and return slices 27 through 75 per file.

Expected input:
    data_dir/
      case_001.nii.gz
      case_002.nii.gz
      ...

Each sample returned by the dataset is:
    image: Tensor with shape [49, H, W]
    meta:  dict with file path and selected slice indices
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Nifti49SliceDataset(Dataset):
    def __init__(
        self,
        data_dir,
        start_slice=27,
        end_slice=75,
        slice_axis=-1,
        normalize=True,
    ):
        self.data_dir = Path(data_dir)
        self.start_slice = start_slice
        self.end_slice = end_slice
        self.slice_axis = slice_axis
        self.normalize = normalize
        self.files = sorted(self.data_dir.rglob("*.nii")) + sorted(
            self.data_dir.rglob("*.nii.gz")
        )

        if not self.files:
            raise FileNotFoundError(f"No .nii or .nii.gz files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        volume = nib.load(str(path)).get_fdata(dtype=np.float32)

        volume = np.moveaxis(volume, self.slice_axis, 0)
        total_slices = volume.shape[0]
        slice_indices = self._select_slice_range(total_slices)
        slices = volume[slice_indices]

        if self.normalize:
            slices = self._minmax_normalize(slices)

        image = torch.from_numpy(slices.astype(np.float32))
        meta = {
            "path": str(path),
            "slice_indices": torch.tensor(slice_indices, dtype=torch.long),
        }
        return image, meta

    def _select_slice_range(self, total_slices):
        if self.start_slice < 0 or self.end_slice < self.start_slice:
            raise ValueError("Slice range must satisfy 0 <= start_slice <= end_slice")
        if self.end_slice >= total_slices:
            raise ValueError(
                f"Need slice {self.end_slice}, but file has only {total_slices} slices"
            )
        return np.arange(self.start_slice, self.end_slice + 1)

    @staticmethod
    def _minmax_normalize(array):
        min_value = array.min()
        max_value = array.max()
        if max_value == min_value:
            return np.zeros_like(array, dtype=np.float32)
        return (array - min_value) / (max_value - min_value)


def create_nii_49_loader(data_dir, batch_size=1, shuffle=True, num_workers=0):
    dataset = Nifti49SliceDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    loader = create_nii_49_loader(
        data_dir="./data",
        batch_size=1,
        shuffle=False,
    )

    for images, meta in loader:
        print(images.shape)
        print(meta["path"])
        print(meta["slice_indices"])
        break
