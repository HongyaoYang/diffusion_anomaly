"""
Train a noised image classifier on BraTS images.

Paper-aligned setting:
- Dataset: BraTS
- Input channels: 4 MRI modalities
- Image size: 256 x 256
- Diffusion steps: 1000
- Noise schedule: linear
- Classifier is trained on noised images
- Used later for classifier guidance during sampling

This script saves the exact classifier + diffusion configuration to JSON,
so guided sampling can use the same parameters.
"""

import argparse
import json
import os
import sys

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import AdamW

sys.path.append("..")
sys.path.append(".")

from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import (
    parse_resume_step_from_filename,
    log_loss_dict,
    visualize,
)

os.environ["OMP_NUM_THREADS"] = "8"


def maybe_create_visdom(args):
    """
    Visdom is useful, but it should not crash training if the server is not running.
    """
    if not args.use_visdom:
        return None, None, None, None

    try:
        from visdom import Visdom

        viz = Visdom(port=args.visdom_port)
        loss_window = viz.line(
            Y=th.zeros((1)).cpu(),
            X=th.zeros((1)).cpu(),
            opts=dict(xlabel="step", ylabel="Loss", title="classification loss"),
        )
        val_window = viz.line(
            Y=th.zeros((1)).cpu(),
            X=th.zeros((1)).cpu(),
            opts=dict(xlabel="step", ylabel="Loss", title="validation loss"),
        )
        acc_window = viz.line(
            Y=th.zeros((1)).cpu(),
            X=th.zeros((1)).cpu(),
            opts=dict(xlabel="step", ylabel="acc", title="accuracy"),
        )
        return viz, loss_window, val_window, acc_window

    except Exception as e:
        print(f"WARNING: Visdom disabled because it failed to initialize: {e}")
        return None, None, None, None


def save_config(args, save_dir):
    """
    Save exact classifier + diffusion training config.
    Guided sampling should match this config.
    """
    os.makedirs(save_dir, exist_ok=True)

    classifier_diffusion_config = {
        k: getattr(args, k)
        for k in classifier_and_diffusion_defaults().keys()
        if hasattr(args, k)
    }

    train_config = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "val_data_dir": args.val_data_dir,
        "noised": args.noised,
        "iterations": args.iterations,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "anneal_lr": args.anneal_lr,
        "batch_size": args.batch_size,
        "microbatch": args.microbatch,
        "schedule_sampler": args.schedule_sampler,
        "resume_checkpoint": args.resume_checkpoint,
        "log_interval": args.log_interval,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
    }

    full_config = {
        "classifier_and_diffusion": classifier_diffusion_config,
        "training": train_config,
    }

    config_path = os.path.join(save_dir, "classifier_train_config.json")
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)

    print(f"\nSaved classifier config to: {config_path}\n")


def print_config(args):
    """
    Print all important parameters into the .out log.
    This allows direct comparison with diffusion_train_config.json.
    """
    print("\n================ CLASSIFIER TRAIN CONFIG ================")

    print("\n[Dataset]")
    print("dataset:", args.dataset)
    print("data_dir:", args.data_dir)
    print("val_data_dir:", args.val_data_dir)

    print("\n[Classifier and diffusion]")
    for k in classifier_and_diffusion_defaults().keys():
        if hasattr(args, k):
            print(f"{k}: {getattr(args, k)}")

    print("\n[Training]")
    print("noised:", args.noised)
    print("iterations:", args.iterations)
    print("lr:", args.lr)
    print("weight_decay:", args.weight_decay)
    print("anneal_lr:", args.anneal_lr)
    print("batch_size:", args.batch_size)
    print("microbatch:", args.microbatch)
    print("schedule_sampler:", args.schedule_sampler)
    print("resume_checkpoint:", args.resume_checkpoint)
    print("log_interval:", args.log_interval)
    print("eval_interval:", args.eval_interval)
    print("save_interval:", args.save_interval)

    print("=========================================================\n")


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    viz, loss_window, val_window, acc_window = maybe_create_visdom(args)

    print_config(args)
    save_config(args, args.save_dir)

    logger.log("creating classifier and diffusion...")

    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys()),
    )

    print("DEBUG classifier_use_fp16:", args.classifier_use_fp16, type(args.classifier_use_fp16))
    print("DEBUG noised:", args.noised, type(args.noised))
    print("DEBUG diffusion.num_timesteps:", diffusion.num_timesteps)

    model.to(dist_util.dev())

    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler,
            diffusion,
            maxt=args.diffusion_steps,
        )
    else:
        schedule_sampler = None

    resume_step = 0

    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        logger.log(
            f"loading classifier checkpoint: {args.resume_checkpoint} at step {resume_step}"
        )
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint,
                map_location=dist_util.dev(),
            )
        )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model,
        use_fp16=args.classifier_use_fp16,
        initial_lg_loss_scale=16.0,
    )

    logger.log("creating data loader...")

    if args.dataset == "brats":
        ds = BRATSDataset(args.data_dir, test_flag=False)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        data = iter(datal)

    elif args.dataset == "chexpert":
        datal = None
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
        print("dataset is chexpert")

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.log("creating optimizer...")

    opt = AdamW(
        mp_trainer.master_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint),
            f"optbratsclass{resume_step:06d}.pt",
        )

        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            opt.load_state_dict(
                dist_util.load_state_dict(
                    opt_checkpoint,
                    map_location=dist_util.dev(),
                )
            )
        else:
            logger.log(f"optimizer checkpoint not found: {opt_checkpoint}")

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, step, prefix="train"):
        if args.dataset == "brats":
            batch, extra, weak_label, seg_label, number = next(data_loader)
            labels = weak_label.long().to(dist_util.dev())
            print("IS BRATS")

        elif args.dataset == "chexpert":
            batch, extra = next(data_loader)
            labels = extra["y"].to(dist_util.dev())
            print("IS CHEXPERT")

        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        print("labels", labels)

        batch = batch.to(dist_util.dev())
        labels = labels.to(dist_util.dev())

        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        batch_correct = 0
        batch_total = 0
        last_losses = None

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            sub_batch = Variable(sub_batch, requires_grad=(prefix != "train"))
            logits = model(sub_batch, timesteps=sub_t)

            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits,
                sub_labels,
                k=1,
                reduction="none",
            )
            losses[f"{prefix}_acc@2"] = compute_top_k(
                logits,
                sub_labels,
                k=2,
                reduction="none",
            )

            print("acc", losses[f"{prefix}_acc@1"])
            log_loss_dict(diffusion, sub_t, losses)

            loss_mean = loss.mean()

            if prefix == "train":
                if viz is not None:
                    viz.line(
                        X=th.ones((1, 1)).cpu() * step,
                        Y=th.Tensor([loss_mean.detach().item()]).unsqueeze(0).cpu(),
                        win=loss_window,
                        name="loss_cls",
                        update="append",
                    )

                if loss_mean.requires_grad:
                    if i == 0:
                        mp_trainer.zero_grad()
                    mp_trainer.backward(loss_mean * len(sub_batch) / len(batch))

            else:
                output_idx = logits[0].argmax()
                output_max = logits[0, output_idx]
                output_max.backward()

                saliency, _ = th.max(sub_batch.grad.data.abs(), dim=1)

                if viz is not None:
                    viz.heatmap(visualize(saliency[0, ...]))
                    viz.image(visualize(sub_batch[0, 0, ...]))
                    viz.image(visualize(sub_batch[0, 1, ...]))

                th.cuda.empty_cache()

            batch_correct += losses[f"{prefix}_acc@1"].sum().item()
            batch_total += len(sub_labels)
            last_losses = losses

        return last_losses, batch_correct, batch_total

    correct = 0
    total = 0

    for step in range(args.iterations - resume_step):
        global_step = step + resume_step

        logger.logkv("step", global_step)
        logger.logkv(
            "samples",
            (global_step + 1) * args.batch_size * dist.get_world_size(),
        )

        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, global_step / args.iterations)

        print("step", global_step)

        try:
            losses, batch_correct, batch_total = forward_backward_log(
                data,
                global_step,
                prefix="train",
            )
        except StopIteration:
            if args.dataset != "brats":
                raise
            data = iter(datal)
            losses, batch_correct, batch_total = forward_backward_log(
                data,
                global_step,
                prefix="train",
            )

        correct += batch_correct
        total += batch_total
        train_acc = correct / max(total, 1)

        if viz is not None:
            viz.line(
                X=th.ones((1, 1)).cpu() * global_step,
                Y=th.Tensor([train_acc]).unsqueeze(0).cpu(),
                win=acc_window,
                name="acc_train",
                update="append",
            )

        mp_trainer.optimize(opt)

        if not global_step % args.log_interval:
            logger.dumpkvs()

        if (
            global_step
            and dist.get_rank() == 0
            and not global_step % args.save_interval
        ):
            logger.log("saving classifier model...")
            save_model(mp_trainer, opt, global_step)

    if dist.get_rank() == 0:
        logger.log("saving classifier model...")
        save_model(mp_trainer, opt, global_step)

    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"modelbratsclass{step:06d}.pt"),
        )
        th.save(
            opt.state_dict(),
            os.path.join(logger.get_dir(), f"optbratsclass{step:06d}.pt"),
        )


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)

    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()

    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def split_microbatches(microbatch, *args):
    bs = len(args[0])

    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)

    else:
        for i in range(0, bs, microbatch):
            yield tuple(
                x[i: i + microbatch] if x is not None else None
                for x in args
            )


def create_argparser():
    defaults = dict(
        # -------------------------
        # Project / dataset
        # -------------------------
        dataset="brats",
        data_dir="./data/brats_2d_z27_75",
        val_data_dir="",
        save_dir="./results/brats_paper_classifier",

        # -------------------------
        # Classifier training
        # -------------------------
        noised=True,

        # Paper/repo style: classifier is usually much shorter than diffusion.
        # If you want to match the author's checkpoint naming, use 20000.
        iterations=20000,

        # Keep classifier lr at 3e-4 unless you specifically want all training lr=1e-4.
        # The diffusion model uses lr=1e-4.
        lr=3e-4,

        weight_decay=0.0,
        anneal_lr=False,
        batch_size=10,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=100,
        eval_interval=1000,
        save_interval=5000,

        # DataLoader
        num_workers=4,

        # Visdom
        use_visdom=True,
        visdom_port=8850,
    )

    defaults.update(classifier_and_diffusion_defaults())

    # -------------------------
    # Paper-aligned diffusion/noising defaults
    # These must match image_train.py.
    # -------------------------
    defaults.update(
        dict(
            image_size=256,
            in_channels=4,
            diffusion_steps=1000,
            noise_schedule="linear",
            classifier_use_fp16=False,
        )
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()