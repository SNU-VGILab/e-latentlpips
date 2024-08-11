import torch
from diffusers import AutoencoderKL

def load_vae(vae_type, device, dtype, requires_grad=False):
    subfolder = "vae"
    if vae_type == "sd1":
        name = "runwayml/stable-diffusion-v1-5"
    elif vae_type == "sdxl":
        name = "stabilityai/stable-diffusion-xl-base-1.0"
        if dtype == torch.float16:
            print("Warning: SDXL vae is not recommended for float16, using https://huggingface.co/madebyollin/sdxl-vae-fp16-fix")
            name = "madebyollin/sdxl-vae-fp16-fix"
            subfolder = None
    elif vae_type == "sd3":
        name = "stabilityai/stable-diffusion-3-medium-diffusers"
    elif vae_type == "flux":
        name = "black-forest-labs/FLUX.1-dev"
        if dtype != torch.bfloat16:
            print("Changing dtype to torch.bfloat16 for FLUX VAE")
            dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")

    vae = AutoencoderKL.from_pretrained(name, subfolder=subfolder)
    vae.requires_grad_(requires_grad)
    vae.to(device=device, dtype=dtype)
    return vae

def scale_vae_output(vae_type, vae, latents):
    if vae_type == "sd1":
        latents = latents * vae.config.scaling_factor
    elif vae_type == "sdxl":
        latents = latents * vae.config.scaling_factor
    elif vae_type == "sd3":
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    elif vae_type == "flux":
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")
    return latents

def unscale_vae_output(vae_type, vae, latents):
    if vae_type == "sd1":
        latents = latents / vae.config.scaling_factor
    elif vae_type == "sdxl":
        latents = latents / vae.config.scaling_factor
    elif vae_type == "sd3":
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    elif vae_type == "flux":
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")
    return latents