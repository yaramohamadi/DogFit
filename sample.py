# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import DiT_models, SiT_models
from download import find_model
from diffusion import create_diffusion
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

import torch.nn as nn

##################################################################################
#                              Guidance control                                  #
##################################################################################
import torch
import torch.nn as nn


class GuidedWrapper(nn.Module):
    """
    Wrapper for DiT or SiT that uses additive embedding guidance,
    configurable via a 3-bit string: zero_init, layer_norm, variance_match.
    """

    def __init__(self, base_model, zero_norm_variance="111", w_dim=1, embed_dim=1152, hidden_dim=128):
        super().__init__()
        self.base_model = base_model
        self.embed_dim = embed_dim

        # Parse boolean flags from zero_norm_variance string
        assert len(zero_norm_variance) == 3, \
            "zero_norm_variance must be a 3-bit string like '101'"
        self.zero_init = zero_norm_variance[0] == "1"
        self.use_layer_norm = zero_norm_variance[1] == "1"
        self.variance_match = zero_norm_variance[2] == "1"

        # Create embedding MLP for guidance scalar
        layers = [
            nn.Linear(w_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        ]
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(embed_dim))
        self.w_embed = nn.Sequential(*layers)

        # Optional zero init
        if self.zero_init:
            for m in self.w_embed.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x, t, y, w=None):
        t_emb = self.base_model.t_embedder(t)                # (B, D)
        y_emb = self.base_model.y_embedder(y, self.training) # (B, D)

        if w is not None:
            w = w.view(-1, 1)  # (B, 1)
            w_emb = self.w_embed(w - 1)  # (B, D)

            if self.variance_match:
                cond_std = (t_emb + y_emb).std(dim=-1, keepdim=True).detach()
                w_emb = w_emb * cond_std * 0.5  # Optional scale

            c = t_emb + y_emb + w_emb
        else:
            c = t_emb + y_emb

        x = self.base_model.x_embedder(x) + self.base_model.pos_embed
        for block in self.base_model.blocks:
            x = block(x, c)
        x = self.base_model.final_layer(x, c)
        x = self.base_model.unpatchify(x)

        if self.base_model.__class__.__name__ == "SiT" and self.base_model.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)





def main(args):
    """
    Run sampling.
    """
    if args.cfg_scale > 1.0:
        assert args.dropout_ratio != 0.0, "cfg_scale > 1.0 requires dropout_ratio != 0.0"
    
    assert not (args.cfg_scale > 1.0 and args.guidance_control > 0), \
    "Cannot use both CFG and DGFT at the same time."
        
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        learn_sigma = args.image_size == 256

    # Load model:
    latent_size = args.image_size // 8
    if args.model.startswith("SiT"):
        print("Loading SiT model...")
        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            class_dropout_prob=args.dropout_ratio,
        ).to(device)
        ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    elif args.model.startswith("DiT"):
        print("Loading DiT model...")
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            class_dropout_prob=args.dropout_ratio,
        ).to(device)
        ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    
    # guidance control
    if args.guidance_control > 0:
        print("wrapping model with GuidedWrapper")
        model = GuidedWrapper(model, args.zero_norm_variance).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

        
    if args.model.startswith("SiT"):
        transport = create_transport(
            args.path_type,
            args.prediction,
            args.loss_weight,
            args.train_eps,
            args.sample_eps
        )
        sampler = Sampler(transport)
        # Assume that the only mode is ODE
        sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
    elif args.model.startswith("DiT"):
        diffusion = create_diffusion(str(args.num_sampling_steps))


    vae_path = f"pretrained_models/sd-vae-ft-{args.vae}"
    if not os.path.exists(vae_path):
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    sample_folder_dir = f"{args.sample_dir}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            model_fn = model.forward_with_cfg
        elif args.guidance_control > 0:
            # Use learnable guidance scale (w)
            w = torch.full((n, 1), fill_value=args.w_dgft, device=device)
            model_kwargs = dict(y=y, w=w)
            model_fn = model.forward
        else:
            # No guidance
            model_kwargs = dict(y=y)
            model_fn = model.forward

        if args.model.startswith("SiT"):
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        elif args.model.startswith("DiT"):
            samples = diffusion.p_sample_loop(
                model_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    dist.destroy_process_group()

all_models = list(SiT_models.keys()) + list(DiT_models.keys())

if __name__ == "__main__":
    def none_or_str(value):
        if value == 'None':
            return None
        return value
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=all_models, default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--dropout-ratio", type=float, default=0.1)
    # Transport arguments
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)
    # Added for guidance control
    parser.add_argument("--guidance-control", type=float, default=0, help="Use learnable guidance scale (w) in the model wrapper")  # DogFit
    parser.add_argument("--w-dgft", type=float, default=1.0, help="Maximum guidance scale") # DogFit
    parser.add_argument("--zero-norm-variance", type=str, default="111") # DogFit
    # ODE arguments
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

    args = parser.parse_args()
    main(args)