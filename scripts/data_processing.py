#!/usr/bin/env python3
"""
Convert raw 3D BraTS NIfTI volumes into patient/z-slice folder format.

Default:
    z = 27 to 75 inclusive
    each z-slice is one 2D training sample
    each sample contains 4 MRI modalities + segmentation mask

Input raw BraTS structure example:
raw_dir/
  BraTS20_Training_001/
    BraTS20_Training_001_flair.nii.gz
    BraTS20_Training_001_t1.nii.gz
    BraTS20_Training_001_t1ce.nii.gz
    BraTS20_Training_001_t2.nii.gz
    BraTS20_Training_001_seg.nii.gz

Output structure:
out_dir/
  training_2d_p001/
    z027/
      brats_train_001_flair_027_w.nii.gz
      brats_train_001_t1_027_w.nii.gz
      brats_train_001_t1ce_027_w.nii.gz
      brats_train_001_t2_027_w.nii.gz
      brats_train_001_seg_027_w.nii.gz
    z028/
      ...
  training_2d_p002/
    z027/
      ...
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES = ["flair", "t1", "t1ce", "t2"]
SEG_NAME = "seg"


def normalize_nonzero(slice_2d: np.ndarray) -> np.ndarray:
    """
    Normalize one 2D MRI slice.

    Background remains 0.
    Non-zero brain region:
        1. z-score normalization
        2. min-max scaling to [0, 1]
    """
    x = slice_2d.astype(np.float32)
    mask = x > 0

    if mask.sum() == 0:
        return x.astype(np.float32)

    vals = x[mask]
    mean = vals.mean()
    std = vals.std()

    if std < 1e-6:
        return x.astype(np.float32)

    x_norm = np.zeros_like(x, dtype=np.float32)
    x_norm[mask] = (x[mask] - mean) / std

    vals_norm = x_norm[mask]
    vmin = vals_norm.min()
    vmax = vals_norm.max()

    if vmax > vmin:
        x_norm[mask] = (x_norm[mask] - vmin) / (vmax - vmin)

    return x_norm.astype(np.float32)


def find_modality_file(patient_dir: Path, modality: str) -> Path:
    """
    Find modality file in one BraTS patient folder.

    Supports both:
        *_flair.nii
        *_flair.nii.gz
    """
    candidates = []
    candidates.extend(sorted(patient_dir.glob(f"*_{modality}.nii.gz")))
    candidates.extend(sorted(patient_dir.glob(f"*_{modality}.nii")))

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {modality} file in {patient_dir}, "
            f"but found {len(candidates)}: {candidates}"
        )

    return candidates[0]


def get_short_patient_id(patient_dir_name: str) -> str:
    """
    Convert:
        BraTS20_Training_001 -> 001
        BraTS2021_00001 -> 00001
    """
    return patient_dir_name.split("_")[-1]


def save_2d_nifti(array_2d: np.ndarray, affine: np.ndarray, out_path: Path):
    """
    Save one 2D array as NIfTI.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(array_2d.astype(np.float32), affine)
    nib.save(img, str(out_path))


def process_one_patient(
    patient_dir: Path,
    out_dir: Path,
    start_slice: int,
    end_slice: int,
    tumor_only: bool,
) -> int:
    """
    Process one patient.

    Returns:
        number of saved z-slice samples for this patient.
    """
    patient_name = patient_dir.name
    short_id = get_short_patient_id(patient_name)

    print("\n" + "-" * 80)
    print(f"Processing patient: {patient_name}")
    print(f"Short ID: {short_id}")

    volumes = {}
    affines = {}

    for mod in MODALITIES + [SEG_NAME]:
        file_path = find_modality_file(patient_dir, mod)
        img = nib.load(str(file_path))

        volumes[mod] = img.get_fdata()
        affines[mod] = img.affine

        print(f"Loaded {mod:5s}: {file_path.name}, shape={volumes[mod].shape}")

    ref_shape = volumes[MODALITIES[0]].shape

    for mod in MODALITIES + [SEG_NAME]:
        if volumes[mod].shape != ref_shape:
            raise ValueError(
                f"Shape mismatch in {patient_name}: "
                f"{mod} has {volumes[mod].shape}, expected {ref_shape}"
            )

    z_dim = ref_shape[2]

    if start_slice < 0 or end_slice >= z_dim:
        raise ValueError(
            f"Invalid z range [{start_slice}, {end_slice}] for {patient_name}. "
            f"Volume z dimension = {z_dim}"
        )

    patient_out_dir = out_dir / f"training_2d_p{short_id}"
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    for z in range(start_slice, end_slice + 1):
        seg_slice = volumes[SEG_NAME][:, :, z].astype(np.float32)

        if tumor_only and seg_slice.max() == 0:
            continue

        slice_out_dir = patient_out_dir / f"z{z:03d}"
        slice_out_dir.mkdir(parents=True, exist_ok=True)

        for mod in MODALITIES:
            img_slice = volumes[mod][:, :, z].astype(np.float32)
            img_slice = normalize_nonzero(img_slice)

            out_name = f"brats_train_{short_id}_{mod}_{z:03d}_w.nii.gz"
            out_path = slice_out_dir / out_name

            save_2d_nifti(img_slice, affines[mod], out_path)

        seg_name = f"brats_train_{short_id}_seg_{z:03d}_w.nii.gz"
        seg_path = slice_out_dir / seg_name

        save_2d_nifti(seg_slice, affines[SEG_NAME], seg_path)

        saved_count += 1

    print(f"Saved z-slice samples for this patient: {saved_count}")

    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw 3D BraTS volumes to patient/z-folder 2D format."
    )

    parser.add_argument(
        "--raw_dir",
        required=True,
        help="Raw BraTS directory containing patient folders.",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--start_slice",
        type=int,
        default=27,
        help="Start z-slice index. Default: 27.",
    )

    parser.add_argument(
        "--end_slice",
        type=int,
        default=75,
        help="End z-slice index, inclusive. Default: 75.",
    )

    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Only process first N patients for debugging.",
    )

    parser.add_argument(
        "--tumor_only",
        action="store_true",
        help="Only save slices whose seg mask contains tumor.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    if args.max_patients is not None:
        patient_dirs = patient_dirs[: args.max_patients]

    print("=" * 80)
    print("BraTS 3D → 2D patient/z-folder preprocessing")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {out_dir}")
    print(f"z range: {args.start_slice} to {args.end_slice}, inclusive")
    print(f"Slices per patient if no filtering: {args.end_slice - args.start_slice + 1}")
    print(f"Patients to process: {len(patient_dirs)}")
    print(f"Tumor only: {args.tumor_only}")
    print("=" * 80)

    total_saved = 0

    for patient_dir in patient_dirs:
        saved = process_one_patient(
            patient_dir=patient_dir,
            out_dir=out_dir,
            start_slice=args.start_slice,
            end_slice=args.end_slice,
            tumor_only=args.tumor_only,
        )
        total_saved += saved

    print("\n" + "=" * 80)
    print("Preprocessing finished.")
    print(f"Total 2D z-slice samples saved: {total_saved}")
    print(f"Total NIfTI files saved: {total_saved * 5}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()