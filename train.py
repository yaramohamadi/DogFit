# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal fine-tuning script for DiT and SiT using PyTorch DDP. This is used as a baseline for our paper "Domain Guidance for Diffusion Models"
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from download import find_model

from models import DiT_models, SiT_models
from diffusion import create_diffusion
from transport import create_transport, Sampler

from diffusers.models import AutoencoderKL

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def load_pretrained_model(model, pretrained_ckpt_path, image_size, tmp_dir="tmp"):
    """
    Load a pre-trained DiT/SiT model for fine-tuning.
    
    Args:
        model: The DiT model instance to load into.
        pretrained_ckpt_path: Optional path to a pre-trained checkpoint. If None, auto-downloads DiT-XL/2.
        image_size: Image size (e.g., 256) to infer checkpoint name if not provided.
        tmp_dir: Temporary directory to save local checkpoint copy.
    
    Returns:
        The model with loaded weights (except y_embedder).
    """
    # Create tmp directory if not exist
    if dist.get_rank() == 0:
        os.makedirs(tmp_dir, exist_ok=True)

    dist.barrier()  # All processes wait here before proceeding

    # Only rank 0 downloads
    if dist.get_rank() == 0:
        if pretrained_ckpt_path is None:
            if args.model.startswith("DiT-XL/2"):
                ckpt_path = pretrained_ckpt_path or f"DiT-XL-2-{image_size}x{image_size}.pt"
            elif args.model.startswith("SiT-XL/2"):
                ckpt_path = pretrained_ckpt_path or f"SiT-XL-2-{image_size}x{image_size}.pt"
        state_dict = find_model(ckpt_path)
        torch.save(state_dict, os.path.join(tmp_dir, "local_pretrained_ckpt.pt"))
    
    dist.barrier()  # Ensure file is fully saved before loading

    # All ranks load from local file
    local_ckpt_path = os.path.join(tmp_dir, "local_pretrained_ckpt.pt")
    state_dict = torch.load(local_ckpt_path, map_location="cpu")

    # Remove incompatible keys (e.g., different number of classes)
    if 'y_embedder.embedding_table.weight' in state_dict:
        del state_dict['y_embedder.embedding_table.weight']
        if dist.get_rank() == 0:
            print("[INFO] Deleted y_embedder.embedding_table.weight to avoid mismatch.")

    # Load the cleaned state_dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if dist.get_rank() == 0:
        print(f"[INFO] Missing keys during loading: {missing_keys}")
        print(f"[INFO] Unexpected keys during loading: {unexpected_keys}")
        print(f"[INFO] Loaded pre-trained weights from {local_ckpt_path}")

    return model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT or SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_dir = f"{args.results_dir}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if args.model in SiT_models:
        model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        class_dropout_prob=args.dropout_ratio,
    )
    elif args.model in DiT_models:
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            class_dropout_prob=args.dropout_ratio,
    )

    # Load pre-trained weights if provided:
    model = load_pretrained_model(model, args.pretrained_ckpt, args.image_size)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])

    if args.model in SiT_models:
        transport = create_transport(
            args.path_type,
            args.prediction,
            args.loss_weight,
            args.train_eps,
            args.sample_eps
        )  # default: velocity; 
        transport_sampler = Sampler(transport)
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    elif args.model in DiT_models:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae_path = f"pretrained_models/sd-vae-ft-{args.vae}"
    if not os.path.exists(vae_path):
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.total_steps} steps...")
    epochs = 999999999 # Set to a large number to avoid epoch-based training

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # DEBUGGGING
            if train_steps == 0 and rank == 0:
                print("=" * 20)
                print(f"Batch x shape: {x.shape}, dtype: {x.dtype}, min: {x.min().item()}, max: {x.max().item()}")
                print(f"Batch y shape: {y.shape}, dtype: {y.dtype}, unique labels: {torch.unique(y)}")
                print("=" * 20)
                # Optionally visualize a few images
                from torchvision.utils import save_image
                save_image(x[:8] * 0.5 + 0.5, f"tmp/sample_batch.png", nrow=4)

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)

            if args.model in SiT_models:
                loss_dict = transport.training_losses(model, x, model_kwargs)
            elif args.model in DiT_models:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT/SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        #"model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        #"opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps > args.total_steps:
                break
        if train_steps > args.total_steps:
                break
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()

all_models = list(SiT_models.keys()) + list(DiT_models.keys())

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=all_models, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    # parser.add_argument("--epochs", type=int, default=1400) # Instead train based on training iterations (Epochs different for each dataset)
    parser.add_argument("--total-steps", type=int, default=24000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4) 
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--pretrained-ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--dropout-ratio", type=float, default=0.1, help="Have null labels or no") # DOG    

    def none_or_str(value):
        if value == 'None':
            return None
        return value

    # For SiT transport models
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

    args = parser.parse_args()
    main(args)