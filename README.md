# Diffusion Anomaly Detection

Extension of diffusion-based anomaly detection methods for brain MRI.

## Project structure

- `guided_diffusion/`: diffusion model implementation
- `scripts/`: training, sampling, preprocessing and visualization scripts
- `train_diffusion.slurm`: diffusion model training job
- `train_classifier.slurm`: classifier training job
- `run_sampling.slurm`: guided sampling job

## Data

Medical imaging datasets and model checkpoints are not included in this repository.

Expected local data directory:

```text
data/
