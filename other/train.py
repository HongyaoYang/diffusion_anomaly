"""
Train the author's guided-diffusion UNet on NIfTI volumes.

This script reads slices [27, 75] from every .nii/.nii.gz file, treats the
49 slices as input channels, and trains a 2D diffusion model.

Example:
    python scripts/train_nii49_diffusion.py \
        --data_dir /path/to/nii_files \
        --image_size 256 \
        --batch_size 1 \
        --steps 100000 \
        --save_dir ./runs/nii49
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_gaussian_diffusion
from guided_diffusion.unet import UNetModel


class Nifti49SliceDataset(Dataset):
    def __init__(
        self,
        data_dir,
        image_size=256,
        start_slice=27,
        end_slice=75,
        slice_axis=-1,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.start_slice = start_slice
        self.end_slice = end_slice
        self.slice_axis = slice_axis
        self.files = sorted(self.data_dir.rglob("*.nii")) + sorted(
            self.data_dir.rglob("*.nii.gz")
        )

        if not self.files:
            raise FileNotFoundError(f"No .nii or .nii.gz files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        volume = nib.load(str(path)).get_fdata(dtype=np.float32)
        volume = np.moveaxis(volume, self.slice_axis, 0)

        if self.end_slice >= volume.shape[0]:
            raise ValueError(
                f"{path} has {volume.shape[0]} slices, cannot read slice {self.end_slice}"
            )

        image = volume[self.start_slice : self.end_slice + 1]
        image = self._normalize(image)
        image = torch.from_numpy(image.astype(np.float32))
        image = self._resize_or_pad(image, self.image_size)

        return image, {"path": str(path)}

    @staticmethod
    def _normalize(image):
        min_value = image.min()
        max_value = image.max()
        if max_value == min_value:
            return np.zeros_like(image, dtype=np.float32)
        image = (image - min_value) / (max_value - min_value)
        return image * 2.0 - 1.0

    @staticmethod
    def _resize_or_pad(image, image_size):
        _, height, width = image.shape
        if height == image_size and width == image_size:
            return image

        image = image.unsqueeze(0)
        if height <= image_size and width <= image_size:
            pad_h = image_size - height
            pad_w = image_size - width
            image = F.pad(
                image,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        else:
            image = F.interpolate(
                image,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
        return image.squeeze(0)


def create_nii49_model(args):
    if args.image_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif args.image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif args.image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif args.image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError("image_size must be one of: 64, 128, 256, 512")

    attention_ds = tuple(args.image_size // int(res) for res in args.attention_resolutions.split(","))
    out_channels = args.in_channels * (2 if args.learn_sigma else 1)

    return UNetModel(
        image_size=args.image_size,
        in_channels=args.in_channels,
        model_channels=args.num_channels,
        out_channels=out_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=attention_ds,
        dropout=args.dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=args.use_checkpoint,
        use_fp16=args.use_fp16,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=-1,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown,
        use_new_attention_order=False,
    )


def save_samples(diffusion, model, args, device, step):
    model.eval()
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            (args.num_samples, args.in_channels, args.image_size, args.image_size),
            clip_denoised=True,
            device=device,
        )
    sample = ((sample.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    torch.save(sample.cpu(), Path(args.save_dir) / f"samples_{step:06d}.pt")
    model.train()


def train(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = Nifti49SliceDataset(
        args.data_dir,
        image_size=args.image_size,
        start_slice=args.start_slice,
        end_slice=args.end_slice,
        slice_axis=args.slice_axis,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = create_nii49_model(args).to(device)
    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=args.diffusion_steps)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1

    data_iter = iter(loader)
    model.train()

    for step in range(start_step, args.steps):
        try:
            batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch, _ = next(data_iter)

        batch = batch.to(device)
        t, weights = schedule_sampler.sample(batch.shape[0], device)
        losses, _ = diffusion.training_losses(model, batch, t, model_kwargs={})
        loss = (losses["loss"] * weights).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            mse = losses["mse"].mean().item()
            print(f"step={step} loss={loss.item():.6f} mse={mse:.6f}")

        if step > 0 and step % args.save_interval == 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                save_dir / f"model_{step:06d}.pt",
            )
            save_samples(diffusion, model, args, device, step)

    torch.save(
        {
            "step": args.steps - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        save_dir / "model_final.pt",
    )


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", default="./runs/nii49")
    parser.add_argument("--resume", default="")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--start_slice", type=int, default=27)
    parser.add_argument("--end_slice", type=int, default=75)
    parser.add_argument("--slice_axis", type=int, default=-1)
    parser.add_argument("--in_channels", type=int, default=49)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=1)

    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", default="linear")
    parser.add_argument("--schedule_sampler", default="uniform")
    parser.add_argument("--learn_sigma", action="store_true")

    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=-1)
    parser.add_argument("--attention_resolutions", default="16,8")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_scale_shift_norm", action="store_true", default=True)
    parser.add_argument("--resblock_updown", action="store_true")
    return parser


if __name__ == "__main__":
    train(create_argparser().parse_args())
