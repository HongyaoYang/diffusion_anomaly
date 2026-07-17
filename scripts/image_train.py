"""
Train a diffusion model on BraTS images.

Paper-aligned setting:
- Dataset: BraTS
- Input channels: 4 MRI modalities
- Image size: 256 x 256
- Diffusion steps: 1000
- Base channels: 128
- Attention resolution: 16
- Learning rate: 1e-4

This script also saves the exact model/diffusion configuration to JSON,
so classifier training and guided sampling can use the same parameters.
"""

import sys
import argparse
import json
import os
from pathlib import Path

import torch as th

sys.path.append("..")
sys.path.append(".")

from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def save_config(args, save_dir):
    """
    Save the exact diffusion/model/training config used in this run.
    Later scripts should read or manually match this config.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_config = {
        k: getattr(args, k)
        for k in model_and_diffusion_defaults().keys()
        if hasattr(args, k)
    }

    train_config = {
        "data_dir": args.data_dir,
        "dataset": args.dataset,
        "schedule_sampler": args.schedule_sampler,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_anneal_steps": args.lr_anneal_steps,
        "batch_size": args.batch_size,
        "microbatch": args.microbatch,
        "ema_rate": args.ema_rate,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "resume_checkpoint": args.resume_checkpoint,
        "use_fp16": args.use_fp16,
        "fp16_scale_growth": args.fp16_scale_growth,
    }

    full_config = {
        "model_and_diffusion": model_config,
        "training": train_config,
    }

    config_path = os.path.join(save_dir, "diffusion_train_config.json")
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)

    print(f"\nSaved diffusion config to: {config_path}\n")


def print_config(args):
    """
    Print all key parameters to the .out log.
    This is essential for checking whether training/sampling configs match.
    """
    print("\n================ DIFFUSION TRAIN CONFIG ================")

    print("\n[Dataset]")
    print("dataset:", args.dataset)
    print("data_dir:", args.data_dir)

    print("\n[Model and diffusion]")
    for k in model_and_diffusion_defaults().keys():
        if hasattr(args, k):
            print(f"{k}: {getattr(args, k)}")

    print("\n[Training]")
    print("schedule_sampler:", args.schedule_sampler)
    print("lr:", args.lr)
    print("weight_decay:", args.weight_decay)
    print("lr_anneal_steps:", args.lr_anneal_steps)
    print("batch_size:", args.batch_size)
    print("microbatch:", args.microbatch)
    print("ema_rate:", args.ema_rate)
    print("log_interval:", args.log_interval)
    print("save_interval:", args.save_interval)
    print("resume_checkpoint:", args.resume_checkpoint)
    print("use_fp16:", args.use_fp16)

    print("========================================================\n")


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    print_config(args)
    save_config(args, args.save_dir)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    print("DEBUG args.class_cond:", args.class_cond, type(args.class_cond))
    print("DEBUG args.learn_sigma:", args.learn_sigma, type(args.learn_sigma))
    print("DEBUG model.num_classes:", getattr(model, "num_classes", None))
    print("DEBUG diffusion.num_timesteps:", diffusion.num_timesteps)

    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler,
        diffusion,
        maxt=args.diffusion_steps,
    )

    logger.log("creating BraTS data loader...")

    ds = BRATSDataset(
        args.data_dir,
        test_flag=False,
    )

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
    ).run_loop()


def create_argparser():
    defaults = dict(
        # -------------------------
        # Project / dataset
        # -------------------------
        dataset="brats",
        data_dir="./data/brats_2d_z27_75",
        save_dir="./results/brats_paper_diffusion",

        # -------------------------
        # Training settings
        # -------------------------
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,

        # Paper uses batch size 10.
        # If GPU memory is not enough, use batch_size=1 and gradient accumulation
        # only if your TrainLoop supports it.
        batch_size=10,
        microbatch=-1,

        ema_rate="0.9999",
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,

        # DataLoader
        num_workers=4,
    )

    defaults.update(model_and_diffusion_defaults())

    # -------------------------
    # Paper-aligned model defaults
    # These override guided_diffusion defaults.
    # -------------------------
    defaults.update(
        dict(
            image_size=256,
            in_channels=4,
            num_channels=128,
            num_res_blocks=2,
            attention_resolutions="16",
            diffusion_steps=1000,
            noise_schedule="linear",
            learn_sigma=False,
            class_cond=True,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_fp16=False,
        )
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()