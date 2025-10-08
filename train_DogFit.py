# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for fine-tuning DiT and SiT through DogFit (Domain Guided Fine-tuning for Transfer Learning of Diffusion Models) using PyTorch DDP. 
The code includes all necessary components of fine-tuning via our method DogFit, including the training loss, guidance control, and training loop.
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
from time import time
import argparse
import logging
import os
import math
import torch.nn as nn
from download import find_model
from models import DiT_models, SiT_models
from diffusion import create_diffusion
from diffusion.gaussian_diffusion import LossType, ModelMeanType, ModelVarType, mean_flat
from transport import create_transport, Sampler, ModelType, path
from diffusers.models import AutoencoderKL
from types import MethodType
from torchvision.utils import save_image

from torchvision.utils import make_grid
import os


############################## Setup logging ######################################


def build_ema_from(model):
    # create the same structure without weights tied to the original
    if isinstance(model, GuidedWrapper):
        # reconstruct wrapper around a deepcopy of the base model
        base_copy = deepcopy(model.base_model)
        ema_wrapped = GuidedWrapper(
            base_copy,
            zero_norm_variance=("1" if model.zero_init else "0")
                              + ("1" if model.use_layer_norm else "0")
                              + ("1" if model.variance_match else "0"),
            w_dim=1, embed_dim=model.embed_dim
        )
        # copy the w_embed weights as well
        ema_wrapped.w_embed.load_state_dict(model.w_embed.state_dict())
        return ema_wrapped
    else:
        return deepcopy(model)




def _unique_step_path(out_dir: str, step: int, prefix: str = "x0_step_", ext: str = ".png") -> str:
    """
    Returns a unique filepath like:
      out_dir/x0_step_0001000.png
      out_dir/x0_step_0001000_1.png  (if the first exists)
      out_dir/x0_step_0001000_2.png  (etc.)
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{prefix}{step:07d}")
    candidate = base + ext
    if not os.path.exists(candidate):
        return candidate
    k = 1
    while True:
        candidate = f"{base}_{k}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1


@torch.no_grad()
def _predict_x0_from_model_output_gauss(diffusion, model_output, x_t, t, model_mean_type):
    """
    Works for common DiT setup:
    - If predicting EPSILON: x0 = (x_t - sqrt(1-ᾱ_t)*ε)/sqrt(ᾱ_t)
    - If predicting START_X: x0 = model_output
    - If predicting PREVIOUS_X: fallback to diffusion helper if available; else return None
    """
    if model_mean_type == ModelMeanType.START_X:
        return model_output

    # gather schedule terms
    alpha_bar = torch.from_numpy(diffusion.alphas_cumprod).to(device=x_t.device, dtype=x_t.dtype)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar[t]).view(-1, 1, 1, 1)

    if model_mean_type == ModelMeanType.EPSILON:
        x0 = (x_t - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
        return x0

    if model_mean_type == ModelMeanType.PREVIOUS_X:
        # if your gaussian_diffusion has _predict_xstart_from_xprev or similar, use it here.
        # otherwise we skip to avoid wrong math.
        return None

    return None


@torch.no_grad()
def save_x0_grid_DiT(
    step, out_dir, model, diffusion, vae, x_start_latents, y, device,
    guidance_control=False, w_minmax=(1.0, 1.0)
):
    """
    Recompute one forward at current step:
      1) sample t and noise, make x_t via q_sample
      2) forward model to get model_output (ε by default)
      3) reconstruct x0, decode with VAE, save grid of the whole batch
    """
    os.makedirs(out_dir, exist_ok=True)

    B = x_start_latents.size(0)
    t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
    noise = torch.randn_like(x_start_latents)
    x_t = diffusion.q_sample(x_start_latents, t, noise=noise)

    model_kwargs = {"y": y}
    if guidance_control:
        wmin, wmax = w_minmax
        w = torch.ones(B, 1, device=device) * ((wmin + wmax) * 0.5)  # just use w=1 (or mid) for stability
        model_kwargs["w"] = w

    model_fn = diffusion._wrap_model(model)
    model_output = model_fn(x_t, t, **model_kwargs)

    # handle learned var case (split C and C)
    if diffusion.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
        C = x_t.shape[1]
        model_output, _ = torch.split(model_output, C, dim=1)

    x0_pred = _predict_x0_from_model_output_gauss(
        diffusion, model_output, x_t, t, diffusion.model_mean_type
    )
    if x0_pred is None:
        return  # skip silently if unsupported mean type

    imgs = vae.decode(x0_pred / 0.18215).sample
    imgs = (imgs.clamp(-1, 1) + 1) * 0.5  # [0,1]

    # make a compact grid respecting batch size
    nrow = int(math.sqrt(B)) if int(math.sqrt(B))**2 == B else min(8, B)
    grid = make_grid(imgs, nrow=nrow, padding=2)
    save_path = _unique_step_path(out_dir, step, prefix="x0_step_", ext=".png")
    save_image(grid, save_path)
    return save_path


@torch.no_grad()
def save_x0_grid_SiT(
    step, out_dir, model, transport, vae, device, y, batch_latents
):
    """
    Preview for SiT/transport (velocity):
      - use the current batch latents as x1 input to transport.sample(x1)
      - plan path to get (t, xt, ut)
      - forward model to get pred velocity
      - reconstruct x0 ≈ xt - sigma_t * pred
      - decode & save grid
    """
    if not isinstance(batch_latents, torch.Tensor):
        return None  # nothing to do

    os.makedirs(out_dir, exist_ok=True)

    # ensure tensors on device
    x1 = batch_latents.detach().to(device, non_blocking=True)
    y  = y.detach().to(device, non_blocking=True)

    # sample path using current batch shape (this is the key change!)
    t, x0, x1 = transport.sample(x1)                       # <— pass x1, NOT None
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)

    # match labels to batch if needed
    if y.shape[0] != xt.shape[0]:
        if y.shape[0] > xt.shape[0]:
            y = y[:xt.shape[0]]
        else:
            # pad by repeating last label
            pad = xt.shape[0] - y.shape[0]
            y = torch.cat([y, y[-1:].repeat(pad)], dim=0)

    model_kwargs = {"y": y}
    pred = model(xt, t, **model_kwargs)

    # x0 reconstruction as in your debug block
    t_expand = t.view(-1, *([1] * (xt.dim() - 1)))
    _, _ = transport.path_sampler.compute_alpha_t(t_expand)   # alpha unused, but fine to keep
    sigma_t, _ = transport.path_sampler.compute_sigma_t(t_expand)
    x0_pred = xt - sigma_t * pred

    imgs = vae.decode(x0_pred / 0.18215).sample
    imgs = (imgs.clamp(-1, 1) + 1) * 0.5

    B = imgs.size(0)
    nrow = int(B**0.5) if int(B**0.5)**2 == B else min(8, B)
    grid = make_grid(imgs, nrow=nrow, padding=2)

    save_path = _unique_step_path(out_dir, step, prefix="x0_step_", ext=".png")
    save_image(grid, save_path)
    return save_path


##################################################################################
#                              Training loss                                     #
##################################################################################


# DogFit training loss function for diffusion models.
def our_training_losses(self, model, x_start, t, model_kwargs=None, noise=None, pretrained_model=None, w_dog=1.0, ema=None, vae=None, guidance_cutoff=False, mg_high=0.75, late_start_iter=0, counter=0):
    """
    Compute training loss with Domain Guided Fine-tuning (DogFit).

    Args:
        model: The diffusion model being trained.
        x_start: Input image tensor [N x C x H x W].
        t: Timesteps for diffusion [N].
        model_kwargs: Extra inputs like class labels or guidance weights.
        noise: Optional Gaussian noise to apply.
        pretrained_model: Source domain pretrained model for guidance.
        w_dog: Guidance strength (scalar or tensor).
        ema: EMA model for current domain.
        vae: Decoder for debugging visualizations.
        guidance_cutoff: If True, apply time-based masking on guidance.
        mg_high: Cutoff threshold (normalized t) for masking.
        late_start_iter: Global step before which no guidance is applied.
        counter: Current training step.

    Returns:
        Dictionary containing loss terms.
    """

    # Debugging function
    def norm_to_01(x):
        """Normalize to [0,1] for visualization."""
        return (x.clamp(-1,1) + 1) / 2
    
    if model_kwargs is None:
        model_kwargs = {}
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_t = self.q_sample(x_start, t, noise=noise)
    terms = {}

    ema_kwargs = dict(model_kwargs)
    y = model_kwargs["y"]
    pretrained_kwargs = {"y": torch.full_like(y, 1000)}

    # Get guidance strength w, (if w is provided in model_kwargs, use it as model input for control; otherwise, use fixed w_dog)
    if model_kwargs.get("w", None) is not None:
        # Extract guidance weight w from model_kwargs
        w = model_kwargs["w"]
        ema_kwargs["w"] = torch.ones_like(w)  # w = 1
    else:
        w = w_dog
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float32, device=x_t.device)
    
    if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
        raise NotImplementedError("Support for KL-based loss is not implemented")
    elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
        
        # If guidance control is enabled, we need to ensure that w = 1 when guidance is not applied.
        if "w" in model_kwargs:
            if pretrained_model is None or ema is None or counter <= late_start_iter:
                model_kwargs["w"] = torch.ones_like(w)  # w = 1
            elif guidance_cutoff:
                # Compute mask based on t < mg_high
                t_norm = t.float() / (self.num_timesteps - 1)  # [B]
                mask = (t_norm < mg_high).float().view(-1, 1)  # [B,1]
                # Set w=1 where guidance is disabled (t >= mg_high)
                w_masked = mask * w + (1 - mask) * torch.ones_like(w)
                model_kwargs["w"] = w_masked
                
        model_output = model(x_t, t, **model_kwargs)

        if pretrained_model is not None and ema is not None and counter > late_start_iter:
            # guidance DogFit
            with torch.no_grad():
                pretrained_output = pretrained_model(x_t, t, **pretrained_kwargs)
                ema_output = ema(x_t, t, **ema_kwargs)

        if self.model_var_type in [
            ModelVarType.LEARNED,
            ModelVarType.LEARNED_RANGE,
        ]:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if pretrained_model is not None and ema is not None and counter > late_start_iter:
                pretrained_output, _ = torch.split(pretrained_output, C, dim=1)
                ema_output, _ = torch.split(ema_output, C, dim=1)

            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            terms["vb"] = self._vb_terms_bpd(
                model=lambda *args, r=frozen_out: r,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
            )["output"]
            if self.loss_type == LossType.RESCALED_MSE:
                # Divide by 1000 for equivalence with initial implementation.
                # Without a factor of 1/1000, the VB term hurts the MSE term.
                terms["vb"] *= self.num_timesteps / 1000.0

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]

        if pretrained_model is not None and ema is not None and counter > late_start_iter:
            # Where the DogFit Happens 

            # Guidance Cut Off
            initial_target = target.clone().detach()
            if guidance_cutoff:
                t_norm = t.float() / (self.num_timesteps - 1)
                mask = (t_norm < mg_high).float().view(-1, 1)  # [16, 1]
                w = w - 1
                w = mask * w  # now w remains [16, 1]
                # The new noise target based on DogFit
                target = target + w.view(-1, 1, 1, 1) * (ema_output.detach() - pretrained_output.detach())
            else:
                # The new noise target based on DogFit
                target = target + (w.view(-1, 1, 1, 1) - 1) * (ema_output.detach() - pretrained_output.detach())


        if pretrained_model is not None and ema is not None and dist.get_rank() == 0 and counter > late_start_iter and counter % 1000 == 0:

            # -----------------------------------------
            # Predict x0 from model and pretrained_model

            # -----------------------------------------
            alpha_bar = torch.from_numpy(self.alphas_cumprod).to(device=x_start.device, dtype=x_start.dtype)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)

            # Reconstruct x0
            x0_model = (x_t - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
            x0_pretrained = (x_t - sqrt_one_minus_alpha_bar_t * pretrained_output) / sqrt_alpha_bar_t

            # Calculate difference for visualization
            x0_diff = (x0_model - x0_pretrained).abs()

            # -----------------------------------------
            # Save all images
            # -----------------------------------------
            save_dir = f"DogFit_debug/{counter:06d}"
            os.makedirs(save_dir, exist_ok=True)

            with torch.no_grad():
                # decode from latents to images
                x_start_decoded = vae.decode(x_start / 0.18215).sample
                x0_model_decoded = vae.decode(x0_model / 0.18215).sample
                x0_pretrained_decoded = vae.decode(x0_pretrained / 0.18215).sample
                x0_diff_decoded = (x0_model_decoded - x0_pretrained_decoded).abs()
                model_noise_decoded = vae.decode(model_output / 0.18215).sample
                initial_noise_decoded = vae.decode(initial_target / 0.18215).sample
                

            # Save normalized images
            save_image(norm_to_01(x_start_decoded),        f"{save_dir}/x_start.png",        nrow=8)
            save_image(norm_to_01(x0_model_decoded),        f"{save_dir}/x0_model.png",       nrow=8)
            save_image(norm_to_01(x0_pretrained_decoded),   f"{save_dir}/x0_pretrained.png",  nrow=8)
            save_image(norm_to_01(x0_diff_decoded),         f"{save_dir}/x0_diff.png",        nrow=8)
            save_image(norm_to_01(model_noise_decoded),         f"{save_dir}/model_noise.png",        nrow=8)
            save_image(norm_to_01(initial_noise_decoded),         f"{save_dir}/initial_noise_decoded.png",        nrow=8)

            print(f"[DEBUG] Saved DogFit debugging images to {save_dir}")
        counter += 1
        
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]
    else:
        raise NotImplementedError(self.loss_type)

    return terms

# This is borrowed from path.py in the transport package.
def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def our_training_losses_transport(
    self,
    model,
    x1,
    model_kwargs=None,
    pretrained_model=None,
    w_dog=1.0,
    ema=None,
    vae=None,
    guidance_cutoff=False,
    mg_high=0.75,
    late_start_iter=0,
    counter=0,
):

    # Debugging function
    def norm_to_01(x):
        """Normalize to [0,1] for visualization."""
        return (x.clamp(-1,1) + 1) / 2
        
    if self.model_type != ModelType.VELOCITY:
        raise NotImplementedError("DogFit is only implemented for ModelType.VELOCITY")

    if model_kwargs is None:
        model_kwargs = {}

    t, x0, x1 = self.sample(x1)
    t, xt, ut = self.path_sampler.plan(t, x0, x1)

    # handle guidance weight w
    ema_kwargs = dict(model_kwargs)
    if model_kwargs.get("w", None) is not None:
        w = model_kwargs["w"]
        ema_kwargs["w"] = torch.ones_like(w)
    else:
        w = w_dog
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float32, device=xt.device)

    model_output = model(xt, t, **model_kwargs)

    B, *_, C = xt.shape
    assert model_output.size() == (B, *xt.size()[1:-1], C)

    if pretrained_model is not None and ema is not None and counter > late_start_iter:
        with torch.no_grad():
            y = model_kwargs["y"]
            # Disable label conditioning for pretrained model (use null class token)
            pretrained_kwargs = {"y": torch.full_like(y, 1000)}
            pretrained_output = pretrained_model(xt, t, **pretrained_kwargs)
            ema_output = ema(xt, t, **ema_kwargs)

        initial_ut = ut.clone().detach()

        if guidance_cutoff:
            t_norm = t
            mask = (t_norm < mg_high).float().view(-1, *([1] * (ut.dim() - 1)))
            w = w - 1
            w = w.view(-1, 1, 1, 1)
            w = mask * w
            ut = ut + w * (ema_output.detach() - pretrained_output.detach())
        else:
            ut = ut + (w.view(-1, *([1] * (ut.dim() - 1))) - 1) * (ema_output.detach() - pretrained_output.detach())

    terms = {"pred": model_output}
    terms["loss"] = mean_flat((model_output - ut) ** 2)

    if pretrained_model is not None and ema is not None and counter > late_start_iter and dist.get_rank() == 0 and counter % 1000 == 0:

        alpha_t, _ = self.path_sampler.compute_alpha_t(expand_t_like_x(t, xt))
        sigma_t, _ = self.path_sampler.compute_sigma_t(expand_t_like_x(t, xt))
        x0_model = xt - sigma_t * model_output
        x0_pretrained = xt - sigma_t * pretrained_output
        x0_diff = (x0_model - x0_pretrained).abs()

        save_dir = f"DogFit_debug/{counter:06d}"
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            x1_dec = vae.decode(x1 / 0.18215).sample
            x0_model_dec = vae.decode(x0_model / 0.18215).sample
            x0_pretrained_dec = vae.decode(x0_pretrained / 0.18215).sample
            x0_diff_dec = (x0_model_dec - x0_pretrained_dec).abs()
            model_noise_decoded = vae.decode(model_output / 0.18215).sample
            initial_noise_decoded = vae.decode(initial_ut / 0.18215).sample

        save_image(norm_to_01(x1_dec), f"{save_dir}/x_start.png", nrow=8)
        save_image(norm_to_01(x0_model_dec), f"{save_dir}/x0_model.png", nrow=8)
        save_image(norm_to_01(x0_pretrained_dec), f"{save_dir}/x0_pretrained.png", nrow=8)
        save_image(norm_to_01(x0_diff_dec), f"{save_dir}/x0_diff.png", nrow=8)
        save_image(norm_to_01(model_noise_decoded), f"{save_dir}/model_noise.png", nrow=8)
        save_image(norm_to_01(initial_noise_decoded), f"{save_dir}/initial_noise.png", nrow=8)

        print(f"[DEBUG] Saved DogFit debug images to {save_dir}")

    return terms



##################################################################################
#                              Guidance control                                  #
##################################################################################
import torch
import torch.nn as nn
import math
import re
import torch
import torch.nn as nn
class GuidedWrapper(nn.Module):
    """
    Wrapper for DiT or SiT that adds a learnable guidance embedding 'w'.
    Robust against deepcopy / DDP / attribute probing.
    """
    def __init__(self, base_model, zero_norm_variance="111", w_dim=1, embed_dim=1152, hidden_dim=128):
        super().__init__()
        # register the wrapped model as a real submodule (goes into self._modules)
        self.base_model = base_model
        self.embed_dim = embed_dim

        assert len(zero_norm_variance) == 3, "zero_norm_variance must be 'abc' bits"
        self.zero_init = (zero_norm_variance[0] == "1")
        self.use_layer_norm = (zero_norm_variance[1] == "1")
        self.variance_match = (zero_norm_variance[2] == "1")

        layers = [
            nn.Linear(w_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        ]
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(embed_dim))
        self.w_embed = nn.Sequential(*layers)

        if self.zero_init:
            for m in self.w_embed.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ----- helpers to access the wrapped base safely -----
    def _base(self):
        b = self._modules.get("base_model", None)  # nn.Module stores submodules here
        if b is None:
            raise AttributeError("base_model")
        return b

    def forward(self, x, t, y, w=None):
        base = self._base()
        t_emb = base.t_embedder(t)                 # (B, D)
        y_emb = base.y_embedder(y, self.training)  # (B, D)

        if w is not None:
            w = w.view(-1, 1)
            w_emb = self.w_embed(w - 1)            # (B, D)
            if self.variance_match:
                cond_std = (t_emb + y_emb).std(dim=-1, keepdim=True).detach()
                w_emb = w_emb * cond_std * 0.5
            c = t_emb + y_emb + w_emb
        else:
            c = t_emb + y_emb

        x = base.x_embedder(x) + base.pos_embed
        for block in base.blocks:
            x = block(x, c)
        x = base.final_layer(x, c)
        x = base.unpatchify(x)

        if base.__class__.__name__ == "SiT" and getattr(base, "learn_sigma", False):
            x, _ = x.chunk(2, dim=1)
        return x

    def __getattr__(self, name):

        # 0) If PyTorch fell through here, first rescue real submodules from _modules.
        if name in ("w_embed", "base_model"):
            m = self._modules.get(name, None)
            if m is not None:
                return m

        # never delegate dunders or core nn.Module plumbing
        if (name.startswith("__") and name.endswith("__")) or name in {
            "training", "_parameters", "_buffers", "_modules",
            "state_dict", "load_state_dict", "named_parameters",
            "named_children", "parameters", "children", "modules",
            "register_buffer", "register_parameter", "to",
            "cpu", "cuda", "eval", "train",
            # block direct access to our own attrs (we’ll get them via _modules when needed)
            "base_model", "w_embed", "embed_dim", "zero_init",
            "use_layer_norm", "variance_match",
        }:
            raise AttributeError(name)
        base = self._modules.get("base_model", None)
        if base is None:
            raise AttributeError(name)
        try:
            return getattr(base, name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        # IMPORTANT: initialize Module internals before attaching submodules
        nn.Module.__init__(new)

        # copy simple flags
        for k in ("embed_dim", "zero_init", "use_layer_norm", "variance_match"):
            setattr(new, k, getattr(self, k))

        # Fetch submodules directly from _modules to avoid our __getattr__
        base = self._modules.get("base_model")
        wemb = self._modules.get("w_embed")

        # Deepcopy the submodules; assigning to attributes registers them in new._modules
        new.base_model = deepcopy(base, memo)
        new.w_embed    = deepcopy(wemb, memo)

        return new


def sample_shifted_exp_custom(size, device, mode="95in1to3"):
    """
    Sample w ~ 1 + Exp(λ), where a specified percentage of samples lie in [1, num].

    Args:
        size: shape of the sample (e.g., (batch_size, 1))
        device: torch device
        mode: string like "95in1to3" or "50in1to2.5"

    Returns:
        torch.Tensor of sampled w values
    """
    match = re.match(r"(\d+)in1to([\d.]+)", mode)
    if not match:
        raise ValueError("Unsupported mode. Use format like '50in1to2' or '95in1to3.5'.")

    percent = float(match.group(1))
    upper = float(match.group(2))

    if not (0 < percent < 100):
        raise ValueError("Percentage must be between 0 and 100.")

    p = percent / 100.0
    interval = upper - 1.0
    if interval <= 0:
        raise ValueError("Upper bound must be greater than 1.")

    lam = -math.log(1.0 - p) / interval

    z = torch.distributions.Exponential(lam).sample(size).to(device)
    return 1.0 + z



#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def load_pretrained_model(model, pretrained_ckpt_path, image_size, tmp_dir="tmp"):
    """
    Load a pre-trained DiT model for fine-tuning.
    
    Args:
        model: The DiT model instance to load into.
        pretrained_ckpt_path: Optional path to a pre-trained checkpoint. If None, auto-downloads DiT-XL/2/SiT-XL/2.
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
    import time
    for attempt in range(50):
        try:
            state_dict = torch.load(local_ckpt_path, map_location="cpu")
            break
        except RuntimeError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    else:
        raise RuntimeError("Failed to load checkpoint after 10 attempts.")

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


# DogFit
# Loads a pre-trained DiT model exactly, including y_embedder.
def load_exact_pretrained_model(model, pretrained_ckpt_path, image_size, tmp_dir="tmp"):
    """
    Loads a pre-trained DiT model exactly, including y_embedder.
    Used for unconditional frozen model in domain-guidance training.
    """
    if dist.get_rank() == 0:
        os.makedirs(tmp_dir, exist_ok=True)
    dist.barrier()

    if dist.get_rank() == 0:
        if args.model.startswith("DiT-XL/2"):
            ckpt_path = pretrained_ckpt_path or f"DiT-XL-2-{image_size}x{image_size}.pt"
        elif args.model.startswith("SiT-XL/2"):
            ckpt_path = pretrained_ckpt_path or f"SiT-XL-2-{image_size}x{image_size}.pt"
        state_dict = find_model(ckpt_path)
        torch.save(state_dict, os.path.join(tmp_dir, "local_pretrained_ckpt.pt"))
    dist.barrier()

    local_ckpt_path = os.path.join(tmp_dir, "local_pretrained_ckpt.pt")
    import time
    for attempt in range(50):
        try:
            state_dict = torch.load(local_ckpt_path, map_location="cpu")
            break
        except RuntimeError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    else:
        raise RuntimeError("Failed to load checkpoint after 10 attempts.")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)  # <-- strict!

    if dist.get_rank() == 0:
        print(f"[INFO] Loaded exact pre-trained model (strict=True) from {local_ckpt_path}")

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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    device = torch.device("cuda", local_rank)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_dir = f"{args.results_dir}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # Logger
        x0_preview_dir = os.path.join(args.results_dir, "_x0_preview")

    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if args.model in SiT_models:
        model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        class_dropout_prob=args.dropout_ratio,  # Domain Guidance dropout ratio
        )
        pretrained_model = SiT_models[args.model](input_size=latent_size, num_classes=1000)
    elif args.model in DiT_models:
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            class_dropout_prob=args.dropout_ratio,  # Domain Guidance dropout ratio
        )
        pretrained_model = DiT_models[args.model](input_size=latent_size, num_classes=1000)
    # Load pre-trained weights if provided:
    model = load_pretrained_model(model, args.pretrained_ckpt, args.image_size)

    # DogFit
    # Load a pre-trained model for domain guidance
    pretrained_model = load_exact_pretrained_model(pretrained_model, args.pretrained_ckpt, args.image_size)  
    requires_grad(pretrained_model, False)
    pretrained_model.eval()
    pretrained_model = pretrained_model.to(device)
    
    # Guidance control:
    if args.guidance_control:
        logger.info("[DogFit] Using guided model wrapper with learnable guidance scale.")
        model = GuidedWrapper(model, args.zero_norm_variance).to(device)
    else:
        logger.info("[DogFit] Using standard model without guidance control.")

    # Note that parameter initialization is done within the DiT constructor

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank])

    if args.model in SiT_models:
        print("LOADING SIT MODEL!")
        transport = create_transport(
            args.path_type,
            args.prediction,
            args.loss_weight,
            args.train_eps,
            args.sample_eps
        )  # default: velocity; 
        transport_sampler = Sampler(transport)
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        transport.training_losses = MethodType(our_training_losses_transport, transport)  # MG
    elif args.model in DiT_models:
        print("LOADING DIT MODEL!")
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        diffusion.training_losses = MethodType(our_training_losses, diffusion) # CG
    vae_path = f"pretrained_models/sd-vae-ft-{args.vae}"
    if not os.path.exists(vae_path):
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    logger.info("[DogFit] Patched diffusion training loss with Domain Guidance (w_DogFit={})".format(args.w_dog))

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
    # update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
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
                save_image(x[:8] * 0.5 + 0.5, f"tmp/sample_batch.png", nrow=4)

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)

            if args.guidance_control:
                # Sample w from Uniform[1.0, args.w_max]
                if args.control_distribution == "uniform":
                    # print(f"Uniform guidance control from {args.w_min} to {args.w_max}")
                    w = torch.empty(x.shape[0], 1, device=device).uniform_(args.w_min, args.w_max)
                else:
                    # print(f"Exponential {args.control_distribution} guidance control from {args.w_min} to {args.w_max}")
                    w = sample_shifted_exp_custom((x.shape[0], 1), device, mode=args.control_distribution)
                sample_shifted_exp_custom
                # print("[DogFit] control distribution:", args.control_distribution)
                # print("[DogFit] Sampled w values:", w.flatten().cpu().numpy())
                model_kwargs["w"] = w

            if args.model in SiT_models:
                loss_dict = transport.training_losses(
                    model,
                    x,
                    model_kwargs,
                    pretrained_model=pretrained_model,
                    ema=ema,
                    vae=vae, # For debugging 
                    w_dog=args.w_dog,
                    guidance_cutoff=args.guidance_cutoff,
                    mg_high=args.mg_high, 
                    late_start_iter=args.late_start_iter,
                    counter=train_steps,
            )


            elif args.model in DiT_models:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                # MG
                # Patch the diffusion training loss to use Domain Guidance
                loss_dict = diffusion.training_losses(
                    model,
                    x,
                    t,
                    model_kwargs,
                    pretrained_model=diffusion._wrap_model(pretrained_model),
                    ema=diffusion._wrap_model(ema),
                    vae=vae, # For debugging 
                    w_dog=args.w_dog,
                    guidance_cutoff=args.guidance_cutoff,
                    mg_high=args.mg_high, 
                    late_start_iter=args.late_start_iter,
                    counter=train_steps,
                )   

            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # update_ema(ema, model.module)

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

            # Save DiT checkpoint:
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

            # Generate x0 preview samples:
            if train_steps % args.log_every == 0 and rank == 0:
                try:
                    if args.model in DiT_models:
                        print("Generating x0 preview with DiT...")
                        save_path = save_x0_grid_DiT(
                            step=train_steps,
                            out_dir=x0_preview_dir,
                            model=ema,                       # prettier; use model.module if you want the raw net
                            diffusion=diffusion,
                            vae=vae,
                            x_start_latents=x,               # current batch latents (already VAE-encoded above)
                            y=y,
                            device=device,
                            guidance_control=bool(args.guidance_control),
                            w_minmax=(args.w_min, args.w_max),
                        )
                    else:
                        print("Generating x0 preview with DiT...")
                        print(x)
                        save_path = save_x0_grid_SiT(
                            step=train_steps,
                            out_dir=x0_preview_dir,
                            model=model.module if isinstance(model, DDP) else model,
                            transport=transport,
                            vae=vae,
                            device=device,
                            y=y,
                            batch_size=x.size(0),
                        )
                    if save_path:
                        logger.info(f"[x0-preview] saved {save_path}")
                except Exception as e:
                    logger.info(f"[x0-preview] skipped due to error: {e}")

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

    def none_or_str(value):
        if value == 'None':
            return None
        return value

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
    parser.add_argument("--w-dog",type=float,default=1.0,help="Domain Guidance strength (w_DogFit)") # DogFit
    parser.add_argument("--guidance-cutoff", type=float, default=0, help="Cutoff for domain guidance") # DogFit
    parser.add_argument("--mg-high", type=float, default=0.75, help="Cutoff for domain guidance") # DogFit
    parser.add_argument("--late-start-iter", type=int, default=0, help="Late start iteration for domain guidance") # DogFit
    parser.add_argument("--dropout-ratio", type=float, default=0.1, help="Have null labels or no") # DogFit
    parser.add_argument("--guidance-control", type=float, default=0, help="Use learnable guidance scale (w) in the model wrapper")  # DOG
    parser.add_argument("--w-max", type=float, default=1.0, help="Maximum guidance scale") # DogFit
    parser.add_argument("--w-min", type=float, default=1.0, help="Maximum guidance scale") # DogFit
    parser.add_argument("--control-distribution", type=str, default="uniform") # DogFit
    parser.add_argument("--zero-norm-variance", type=str, default="111") # DogFit
    # For SiT transport models
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

    args = parser.parse_args()
    main(args)