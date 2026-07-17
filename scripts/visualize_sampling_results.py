"""Create offline BraTS reconstruction/anomaly-map figures from sampling outputs.

The script reproduces the deterministic selection used by
classifier_sample_known.py: it walks the BraTS dataset in loader order and
keeps the first N slices whose segmentation mask contains tumour.  This lets
the saved samples.npy and anomaly_maps.npy be compared with their original
MRI and ground-truth segmentation without rerunning diffusion sampling.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Running ``python scripts/visualize_sampling_results.py`` puts only the
# scripts directory on sys.path. Add the repository root for guided_diffusion.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion.bratsloader import BRATSDataset


MODALITIES = ("T1", "T1ce", "T2", "FLAIR")


def to_numpy(value):
    """Convert a torch tensor (or numpy array) to a NumPy array."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def normalize(image):
    """Safely map an image to [0, 1] for display."""
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def load_selected_slices(data_dir, count):
    """Return the first ``count`` tumour-containing slices in sampling order."""
    dataset = BRATSDataset(data_dir, test_flag=False)
    selected = []

    for index in range(len(dataset)):
        image, _, weak_label, segmentation, sample_id = dataset[index]
        if int(weak_label) != 1:
            continue

        mask = np.zeros((256, 256), dtype=np.float32)
        mask[8:-8, 8:-8] = to_numpy(segmentation)[0]
        selected.append((sample_id, to_numpy(image), mask))
        if len(selected) == count:
            return selected

    raise RuntimeError(
        f"Found only {len(selected)} tumour-containing slices, but {count} results were supplied."
    )


def save_sample_figure(sample_id, original, reconstruction, anomaly_map, mask, output_path):
    """Save one five-column, four-modality comparison figure."""
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), constrained_layout=True)
    columns = ("Original", "Reconstruction", "Absolute error", "Anomaly map", "Ground truth")
    anomaly_limit = float(np.percentile(anomaly_map, 99))
    anomaly_limit = anomaly_limit if anomaly_limit > 0 else 1.0

    for row, modality in enumerate(MODALITIES):
        absolute_error = np.abs(original[row] - reconstruction[row])
        panels = (
            (normalize(original[row]), "gray", 0, 1),
            (normalize(reconstruction[row]), "gray", 0, 1),
            (normalize(absolute_error), "magma", 0, 1),
            (anomaly_map, "hot", 0, anomaly_limit),
            (mask, "viridis", 0, 1),
        )
        for col, (image, cmap, vmin, vmax) in enumerate(panels):
            axes[row, col].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(columns[col])
        axes[row, 0].set_ylabel(modality, rotation=0, labelpad=32, va="center")

    fig.suptitle(sample_id, fontsize=14)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_overview(selected, anomaly_maps, output_path):
    """Save a compact overview: anomaly map beside its ground-truth mask."""
    rows = len(selected)
    fig, axes = plt.subplots(rows, 2, figsize=(7, 3 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    anomaly_limit = float(np.percentile(anomaly_maps, 99))
    anomaly_limit = anomaly_limit if anomaly_limit > 0 else 1.0

    for row, ((sample_id, _, mask), anomaly_map) in enumerate(zip(selected, anomaly_maps)):
        axes[row, 0].imshow(anomaly_map, cmap="hot", vmin=0, vmax=anomaly_limit)
        axes[row, 0].set_title(f"{sample_id}: anomaly map")
        axes[row, 1].imshow(mask, cmap="viridis", vmin=0, vmax=1)
        axes[row, 1].set_title("ground-truth segmentation")
        for axis in axes[row]:
            axis.axis("off")

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="BraTS 2D data directory used for sampling.")
    parser.add_argument("--results_dir", required=True, help="Directory containing samples.npy and anomaly_maps.npy.")
    parser.add_argument("--output_dir", default=None, help="Directory for generated PNGs; defaults to <results_dir>/visualizations.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = np.load(results_dir / "samples.npy")
    anomaly_maps = np.load(results_dir / "anomaly_maps.npy")
    if samples.ndim != 4 or samples.shape[1:] != (4, 256, 256):
        raise ValueError(f"Expected samples shape [N, 4, 256, 256], received {samples.shape}.")
    if anomaly_maps.shape != (samples.shape[0], 256, 256):
        raise ValueError(
            "Expected anomaly_maps shape "
            f"[{samples.shape[0]}, 256, 256], received {anomaly_maps.shape}."
        )

    selected = load_selected_slices(args.data_dir, samples.shape[0])
    manifest_path = output_dir / "sample_manifest.csv"
    with manifest_path.open("w", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=("sample_index", "sample_id", "figure"))
        writer.writeheader()
        for index, ((sample_id, original, mask), reconstruction, anomaly_map) in enumerate(
            zip(selected, samples, anomaly_maps)
        ):
            figure_name = f"{index:02d}_{sample_id}.png"
            save_sample_figure(sample_id, original, reconstruction, anomaly_map, mask, output_dir / figure_name)
            writer.writerow({"sample_index": index, "sample_id": sample_id, "figure": figure_name})

    save_overview(selected, anomaly_maps, output_dir / "overview_anomaly_vs_ground_truth.png")
    print(f"Saved {samples.shape[0]} comparison figures to: {output_dir}")
    print(f"Saved sample manifest to: {manifest_path}")


if __name__ == "__main__":
    main()