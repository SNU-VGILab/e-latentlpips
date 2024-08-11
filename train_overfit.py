import os
import argparse
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import lpips
import wandb
import torch

from elatentlpips import ELatentLPIPS
from utils import load_vae, scale_vae_output, unscale_vae_output

def parse_args():
    parser = argparse.ArgumentParser(description="Train (single) sample overfitting")
    parser.add_argument("--n_images", type=int, default=1, help="Number of images to use = batch size")
    parser.add_argument("--n_steps", type=int, default=100000, help="Number of steps to train")
    parser.add_argument("--output_dir", type=str, default="outputs/debug", help="Output directory")
    parser.add_argument("--output_freq", type=int, default=1000, help="Output frequency")

    parser.add_argument("--aug_type", type=str, default="bg+cutout", help="augmentation to use (only applied to latent lpips)",)
    parser.add_argument("--aug_strength", type=float, default=1., help="strength of augmentation")

    parser.add_argument("--lpips_domain", type=str, default="latent", choices=["latent", "pixel"], help="whether to use real LPIPS or latent LPIPS")
    parser.add_argument("--model_type", type=str, default="VGG16_Latent_GN", choices=["VGG16_Latent", "VGG16_Latent_BN", "VGG16_Latent_GN"], help="model type to use")
    parser.add_argument("--pretrained_latent_lpips_path", type=str, default="/root/data2/e-latentlpips/latent_lpips_vgg16_gn_sd1/checkpoint-ep=0/model.safetensors", help="path to pretrained latent LPIPS model")
    
    parser.add_argument("--vae_type", type=str, default="sd1", choices=["sd1", "sdxl", "sd3", "flux"], help="which VAE to use")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--cuda_id", type=int, default=0, help="cuda device id")
    return parser.parse_args()

def main(args):
    device = f"cuda:{args.cuda_id}"
    input_images = [Image.open(os.path.join("test_images", f"{i+1}.png")).convert("RGB") for i in range(args.n_images)] # already of size 512x512
    input_images = [np.array(image).transpose(2, 0, 1) for image in input_images]
    input_images = 2 * (torch.tensor(np.stack(input_images)) / 255.0) - 1

    vae = load_vae(args.vae_type, device, torch.float32, requires_grad=False)

    target_tensors = input_images.to(device=device)
    temp = vae.encode(target_tensors.to(vae.dtype)).latent_dist.sample()
    source_tensors = torch.nn.Parameter(torch.randn_like(temp).float(), requires_grad=True)
    optimizer = torch.optim.Adam([source_tensors], lr=args.lr, betas=(0.9, 0.999))

    if args.lpips_domain == "pixel":
        vae.decoder.requires_grad_(True)
        lpips_model = lpips.LPIPS(net='vgg').to(device=device)    
    else:
        target_tensors = vae.encode(target_tensors.to(vae.dtype)).latent_dist.sample()
        target_tensors = scale_vae_output(args.vae_type, vae, target_tensors).float()
        lpips_model = ELatentLPIPS(backbone_type=args.model_type, backbone_in_channels=4 if args.vae_type in ["sd1", "sdxl"] else 16,
                                   pretrained_latent_lpips_path=args.pretrained_latent_lpips_path,
                                   aug_type=args.aug_type, aug_strength=args.aug_strength).to(device=device)

    wandb.init(project="E-Latent LPIPS Overfit", config=args)
    os.makedirs(args.output_dir, exist_ok=True)

    progress_bar = tqdm(range(args.n_steps), dynamic_ncols=True)
    for step in range(args.n_steps):
        if args.lpips_domain == "pixel":
            unscaled_source_tensors = unscale_vae_output(args.vae_type, vae, source_tensors).float()
            decoded_source_tensors = vae.decode(unscaled_source_tensors)["sample"]
            loss = lpips_model(target_tensors, decoded_source_tensors)
        else:
            loss = lpips_model(target_tensors, source_tensors)
        
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        
        wandb.log({"Loss": loss.mean().item()}, step=step)
        progress_bar.set_description(f"Loss: {loss.mean().item():.4f}")
        progress_bar.update()

        if step % args.output_freq == 0:
            with torch.no_grad():
                unscaled_source_tensors = unscale_vae_output(args.vae_type, vae, source_tensors).float()
                output_images = vae.decode(unscaled_source_tensors.to(vae.dtype))["sample"] # [-1, 1]
                output_images = (output_images + 1) / 2
                output_images = (output_images * 255).clamp(0, 255).to("cpu", torch.uint8).numpy()
                for i, image in enumerate(output_images):
                    Image.fromarray(image.transpose(1, 2, 0)).save(os.path.join(args.output_dir, f"{step:06d}_{i}.png"))
                    if i == 0:
                        wandb.log({"Image": wandb.Image(Image.fromarray(image.transpose(1, 2, 0)))}, step=step)

if __name__ == "__main__":
    args = parse_args()
    main(args)