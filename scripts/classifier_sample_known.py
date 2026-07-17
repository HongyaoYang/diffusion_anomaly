"""
Guided sampling / reconstruction for BraTS anomaly detection.

Paper-aligned setting:
- Dataset: BraTS
- Input channels: 4 MRI modalities
- Image size: 256 x 256
- Diffusion steps: 1000
- Noise schedule: linear
- Noise level L: 500
- Classifier scale s: 100
- Target class: healthy class 0

This script:
1. Loads a trained diffusion UNet.
2. Loads a trained noised image classifier.
3. Takes diseased BraTS images only.
4. Adds noise to level L.
5. Reconstructs with classifier guidance toward healthy class.
6. Computes anomaly map as |original - reconstruction|.
7. Saves sampling config and outputs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

sys.path.append("..")
sys.path.append(".")

from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def visualize(img, eps=1e-8):
    """
    Normalize tensor/image to [0, 1] for visualization.
    Safe when image is constant.
    """
    _min = img.min()
    _max = img.max()
    return (img - _min) / (_max - _min + eps)


def maybe_create_visdom(args):
    """
    Visdom is optional. Sampling should not crash if Visdom is unavailable.
    """
    if not args.use_visdom:
        return None

    try:
        from visdom import Visdom

        viz = Visdom(port=args.visdom_port)
        return viz

    except Exception as e:
        print(f"WARNING: Visdom disabled because it failed to initialize: {e}")
        return None


def save_config(args, save_dir):
    """
    Save exact sampling configuration.
    This helps ensure train / classifier / sample configs are traceable.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_config = {
        k: getattr(args, k)
        for k in model_and_diffusion_defaults().keys()
        if hasattr(args, k)
    }

    classifier_config = {
        k: getattr(args, k)
        for k in classifier_defaults().keys()
        if hasattr(args, k)
    }

    sampling_config = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "model_path": args.model_path,
        "classifier_path": args.classifier_path,
        "clip_denoised": args.clip_denoised,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "use_ddim": args.use_ddim,
        "classifier_scale": args.classifier_scale,
        "noise_level": args.noise_level,
        "target_class": args.target_class,
        "save_dir": args.save_dir,
    }

    full_config = {
        "model_and_diffusion": model_config,
        "classifier": classifier_config,
        "sampling": sampling_config,
    }

    config_path = os.path.join(save_dir, "classifier_sample_config.json")
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)

    print(f"\nSaved sampling config to: {config_path}\n")


def print_config(args):
    """
    Print key config into .out log.
    """
    print("\n================ GUIDED SAMPLING CONFIG ================")

    print("\n[Dataset]")
    print("dataset:", args.dataset)
    print("data_dir:", args.data_dir)

    print("\n[Paths]")
    print("model_path:", args.model_path)
    print("classifier_path:", args.classifier_path)
    print("save_dir:", args.save_dir)

    print("\n[Model and diffusion]")
    for k in model_and_diffusion_defaults().keys():
        if hasattr(args, k):
            print(f"{k}: {getattr(args, k)}")

    print("\n[Classifier]")
    for k in classifier_defaults().keys():
        if hasattr(args, k):
            print(f"{k}: {getattr(args, k)}")

    print("\n[Sampling]")
    print("clip_denoised:", args.clip_denoised)
    print("num_samples:", args.num_samples)
    print("batch_size:", args.batch_size)
    print("use_ddim:", args.use_ddim)
    print("classifier_scale:", args.classifier_scale)
    print("noise_level:", args.noise_level)
    print("target_class:", args.target_class)

    print("========================================================\n")


def check_required_paths(args):
    if not args.model_path:
        raise ValueError("model_path is empty. Please provide --model_path.")

    if not args.classifier_path:
        raise ValueError("classifier_path is empty. Please provide --classifier_path.")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model_path does not exist: {args.model_path}")

    if not os.path.exists(args.classifier_path):
        raise FileNotFoundError(f"classifier_path does not exist: {args.classifier_path}")


def compare_state_dict_shapes(module, state_dict, name):
    """
    Print mismatch summary before load_state_dict.
    This makes architecture mismatch easier to diagnose.
    """
    module_sd = module.state_dict()

    missing = []
    unexpected = []
    mismatch = []

    for k in module_sd.keys():
        if k not in state_dict:
            missing.append(k)
        elif module_sd[k].shape != state_dict[k].shape:
            mismatch.append((k, tuple(module_sd[k].shape), tuple(state_dict[k].shape)))

    for k in state_dict.keys():
        if k not in module_sd:
            unexpected.append(k)

    print(f"\n================ {name} CHECKPOINT CHECK ================")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    print("Shape mismatches:", len(mismatch))

    if mismatch:
        print("\nFirst 20 shape mismatches:")
        for item in mismatch[:20]:
            print(item)

    if missing:
        print("\nFirst 20 missing keys:")
        for item in missing[:20]:
            print(item)

    if unexpected:
        print("\nFirst 20 unexpected keys:")
        for item in unexpected[:20]:
            print(item)

    print("========================================================\n")

    if missing or unexpected or mismatch:
        raise RuntimeError(
            f"{name} checkpoint does not match the current architecture. "
            "Check image_size / in_channels / num_channels / num_res_blocks / "
            "attention_resolutions / class_cond / learn_sigma / classifier settings."
        )


def main():
    args = create_argparser().parse_args()

    check_required_paths(args)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    viz = maybe_create_visdom(args)

    print_config(args)
    save_config(args, args.save_dir)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    print("DEBUG model diffusion.num_timesteps:", diffusion.num_timesteps)
    print("DEBUG args.class_cond:", args.class_cond, type(args.class_cond))
    print("DEBUG args.learn_sigma:", args.learn_sigma, type(args.learn_sigma))

    if args.dataset == "brats":
        # Anomaly reconstruction needs the segmentation mask to identify
        # diseased slices.  test_flag=True sets every weak_label to 0, which
        # would make the loop below skip every sample.
        ds = BRATSDataset(args.data_dir, test_flag=False)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    elif args.dataset == "chexpert":
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
        datal = iter(data)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.log("loading diffusion model checkpoint...")

    model_state = dist_util.load_state_dict(args.model_path, map_location="cpu")
    compare_state_dict_shapes(model, model_state, name="DIFFUSION MODEL")

    model.load_state_dict(model_state)
    model.to(dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()

    logger.log("loading classifier...")

    classifier = create_classifier(
        **args_to_dict(args, classifier_defaults().keys())
    )

    classifier_state = dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    compare_state_dict_shapes(classifier, classifier_state, name="CLASSIFIER")

    classifier.load_state_dict(classifier_state)
    classifier.to(dist_util.dev())

    if args.classifier_use_fp16:
        classifier.convert_to_fp16()

    classifier.eval()

    print("loaded classifier")

    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print("pmodel", p1, "pclass", p2)

    def cond_fn(x, t, y=None):
        """
        Classifier guidance:
        grad_x log p(y | x_t, t)

        Here y is target_class, normally 0 = healthy.
        """
        assert y is not None

        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)

            selected = log_probs[th.arange(len(logits), device=logits.device), y.view(-1)]
            grad = th.autograd.grad(selected.sum(), x_in)[0]

            return grad, grad * args.classifier_scale

    def model_fn(x, t, y=None):
        """
        Diffusion model function.
        If class_cond=True, pass target class y into the diffusion UNet.
        """
        if args.class_cond:
            assert y is not None
            return model(x, t, y)

        return model(x, t, None)

    logger.log("sampling...")

    all_images = []
    all_labels = []
    all_anomaly_maps = []

    processed = 0

    for img in datal:
        if processed >= args.num_samples:
            break

        model_kwargs = {}

        if args.dataset == "brats":
            batch = img[0]
            weak_label = img[2]
            seg_label = img[3]
            number = img[4][0]

            print("img", batch.shape, "weak_label", weak_label, "number", number)

            # Use only diseased images as input, following anomaly reconstruction.
            # For batch_size=1 this is straightforward.
            # For batch_size>1, this simple version skips the whole batch if no disease exists.
            if th.as_tensor(weak_label).sum().item() == 0:
                continue

            labelmask = th.where(seg_label > 0, 1, 0)

            if viz is not None:
                viz.image(visualize(batch[0, 0, ...]), opts=dict(caption="img input 0"))
                viz.image(visualize(batch[0, 1, ...]), opts=dict(caption="img input 1"))
                viz.image(visualize(batch[0, 2, ...]), opts=dict(caption="img input 2"))
                viz.image(visualize(batch[0, 3, ...]), opts=dict(caption="img input 3"))
                viz.image(visualize(labelmask[0, ...]), opts=dict(caption="ground truth"))

        elif args.dataset == "chexpert":
            batch = img[0]
            number = img[1]["path"]
            print("img", batch.shape, "number", number)

            if viz is not None:
                viz.image(visualize(batch[0, ...]), opts=dict(caption="img input"))

        # Explicit target class.
        # In this project, target_class=0 means healthy reconstruction.
        if args.class_cond:
            classes = th.full(
                size=(batch.shape[0],),
                fill_value=args.target_class,
                dtype=th.long,
                device=dist_util.dev(),
            )
            model_kwargs["y"] = classes
            print("target y", model_kwargs["y"])
        else:
            classes = th.full(
                size=(batch.shape[0],),
                fill_value=args.target_class,
                dtype=th.long,
                device=dist_util.dev(),
            )

        sample_fn = (
            diffusion.ddim_sample_loop_known
            if args.use_ddim
            else diffusion.p_sample_loop_known
        )

        print("sample_fn", sample_fn)

        shape = (
            batch.shape[0],
            args.in_channels,
            args.image_size,
            args.image_size,
        )

        if th.cuda.is_available():
            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)
            start.record()

        sample, x_noisy, org = sample_fn(
            model_fn,
            shape,
            img,
            org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level,
        )

        if th.cuda.is_available():
            end.record()
            th.cuda.synchronize()
            th.cuda.current_stream().synchronize()
            print("sampling time ms", start.elapsed_time(end))

        if args.dataset == "brats":
            anomaly_map = th.abs(org[0, : args.in_channels, ...] - sample[0, ...]).sum(dim=0)

            if viz is not None:
                viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output 0"))
                viz.image(visualize(sample[0, 1, ...]), opts=dict(caption="sampled output 1"))
                viz.image(visualize(sample[0, 2, ...]), opts=dict(caption="sampled output 2"))
                viz.image(visualize(sample[0, 3, ...]), opts=dict(caption="sampled output 3"))
                viz.heatmap(visualize(anomaly_map), opts=dict(caption="anomaly map"))

        elif args.dataset == "chexpert":
            anomaly_map = th.abs(
                visualize(org[0, 0, ...]) - visualize(sample[0, 0, ...])
            )

            if viz is not None:
                viz.image(visualize(sample[0, ...]), opts=dict(caption="sampled output"))
                viz.heatmap(np.flipud(np.array(anomaly_map.cpu())), opts=dict(caption="diff"))

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([s.cpu().numpy() for s in gathered_samples])

        gathered_maps = [
            th.zeros_like(anomaly_map.unsqueeze(0))
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_maps, anomaly_map.unsqueeze(0))
        all_anomaly_maps.extend([m.cpu().numpy() for m in gathered_maps])

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([lab.cpu().numpy() for lab in gathered_labels])

        processed += batch.shape[0]

    if len(all_images) == 0:
        raise RuntimeError(
            "No samples were generated. Check that the dataset contains "
            "diseased slices with non-empty segmentation masks."
        )

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    anomaly_arr = np.concatenate(all_anomaly_maps, axis=0)
    anomaly_arr = anomaly_arr[: args.num_samples]

    os.makedirs(args.save_dir, exist_ok=True)

    samples_path = os.path.join(args.save_dir, "samples.npy")
    anomaly_path = os.path.join(args.save_dir, "anomaly_maps.npy")

    np.save(samples_path, arr)
    np.save(anomaly_path, anomaly_arr)

    print("Saved samples to:", samples_path)
    print("Saved anomaly maps to:", anomaly_path)

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

        labels_path = os.path.join(args.save_dir, "labels.npy")
        np.save(labels_path, label_arr)
        print("Saved labels to:", labels_path)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        # -------------------------
        # Project / dataset
        # -------------------------
        dataset="brats",
        data_dir="./data/brats_2d_z27_75",
        save_dir="./results/brats_paper_sampling",

        # -------------------------
        # Paths
        # -------------------------
        model_path="",
        classifier_path="",

        # -------------------------
        # Sampling settings
        # -------------------------
        clip_denoised=True,
        num_samples=10,
        batch_size=1,

        # Paper uses DDIM-like deterministic reconstruction.
        use_ddim=True,

        # Paper-style guidance setting.
        classifier_scale=100,
        noise_level=500,

        # 0 = healthy target class.
        target_class=0,

        # DataLoader
        num_workers=4,

        # Visdom
        use_visdom=True,
        visdom_port=8850,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())

    # -------------------------
    # Paper-aligned diffusion defaults
    # Must match image_train.py.
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

    # -------------------------
    # Paper-aligned classifier defaults
    # Must match classifier_train.py.
    # -------------------------
    defaults.update(
        dict(
            classifier_use_fp16=False,
        )
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
