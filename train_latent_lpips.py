import os
import argparse
import math
import logging
import numpy as np
from functools import partial
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import diffusers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from datasets import load_from_disk
from elatentlpips import LatentLPIPS
from utils import load_vae, scale_vae_output

def parse_args():
    parser = argparse.ArgumentParser(description="Train latent lpips model")
    parser.add_argument("--eval_only", action="store_true", help="Whether to only evaluate the model")
    parser.add_argument("--eval_model_path", type=str, default="checkpoints/latent_lpips_vgg16_gn_sd1/checkpoint-ep=9", help="path to the model checkpoint for evaluation")

    parser.add_argument("--dataset_name", type=str, default="/root/data/dataset/vision_general/bapps/2afc/2afc_dataset_train", help="path to the training dataset")
    parser.add_argument("--val_dataset_name", type=str, default="/root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val", help="path to the validation dataset")
    parser.add_argument("--vae_type", type=str, default="sd1", choices=["sd1", "sdxl", "sd3", "flux"], help="type of VAE to use")

    parser.add_argument("--model_type", type=str, default="VGG16_Latent_GN", choices=["VGG16_Latent", "VGG16_Latent_BN", "VGG16_Latent_GN"], help="model type to use")
    parser.add_argument("--pretrained_backbone_path", type=str, default="model.safetensors", help="path to the pretrained backbone model")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size for training (effective batch size will be batch_size * gradient_accumulation_steps * num_processes")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="number of workers for the dataloader")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs to train, last half of the epochs will be used with lr decay")
    parser.add_argument("--output_dir", type=str, default="checkpoints/latent_lpips_vgg16_gn_sd1", help="Directory for model predictions and checkpoints")
    parser.add_argument("--seed", type=int, default=4727, help="Random seed for reproducibility")

    args = parser.parse_args()
    return args

def preprocess(examples, resolution=(64, 64)):
    ref_images = [np.array(Image.open(img).convert("RGB").resize(resolution, Image.BICUBIC)).transpose(2,0,1) for img in examples["ref"]]
    p0_images = [np.array(Image.open(img).convert("RGB").resize(resolution, Image.BICUBIC)).transpose(2,0,1) for img in examples["p0"]]
    p1_images = [np.array(Image.open(img).convert("RGB").resize(resolution, Image.BICUBIC)).transpose(2,0,1) for img in examples["p1"]]
    judges = [np.load(judge).reshape((1, 1, 1,)) for judge in examples["judge"]]

    examples["ref_tensors"] = 2 * (torch.tensor(np.stack(ref_images)) / 255) - 1
    examples["p0_tensors"] = 2 * (torch.tensor(np.stack(p0_images)) / 255) - 1
    examples["p1_tensors"] = 2 * (torch.tensor(np.stack(p1_images)) / 255) - 1
    examples["judge_tensors"] = torch.tensor(np.stack(judges))

    return examples

def collate_fn(examples):
    ref_images = torch.stack([example["ref_tensors"] for example in examples])
    p0_images = torch.stack([example["p0_tensors"] for example in examples])
    p1_images = torch.stack([example["p1_tensors"] for example in examples])
    judges = torch.stack([example["judge_tensors"] for example in examples])
    return {"ref": ref_images, "p0": p0_images, "p1": p1_images, "judge": judges}

def compute_accuracy(d0,d1,judge):
    d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
    judge_per = judge.cpu().numpy().flatten()
    return d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)

class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2. # again to [0,1]
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

def main(args):
    logger = get_logger(__name__, log_level="INFO")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="wandb",
        project_config=accelerator_project_config,
        step_scheduler_with_optimizer=False
    )
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    vae = load_vae(args.vae_type, device=accelerator.device, dtype=torch.float16, requires_grad=False)

    lpips = LatentLPIPS(backbone_type=args.model_type, pretrained_backbone_path=args.pretrained_backbone_path, use_dropout=True, eval_mode=False)
    rankLoss = BCERankingLoss()
    trainable_params = list(lpips.parameters()) + list(rankLoss.parameters())

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, betas=(0.5, 0.999))

    train_dataset = load_from_disk(args.dataset_name)
    valid_dataset = load_from_disk(args.val_dataset_name)
    with accelerator.main_process_first():
        train_dataset = train_dataset.with_transform(partial(preprocess, resolution=(512, 512)))
        valid_dataset = valid_dataset.with_transform(partial(preprocess, resolution=(512, 512)))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )

    lpips, rankLoss, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        lpips, rankLoss, optimizer, train_dataloader, valid_dataloader
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("E-Latent LPIPS", config=vars(args))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    total_batch_size = args.batch_size * accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Num Update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {num_update_steps_per_epoch * args.num_epochs}")
    logger.info(f"  Total trainable parameters = {sum(p.numel() for p in trainable_params if p.requires_grad)}")
    first_epoch = 0
    global_step = 0

    for epoch in range(first_epoch, args.num_epochs):
        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
        progress_bar.set_description(f"Epoch: {epoch}/{args.num_epochs}")
        lpips.train()
        for step, batch in enumerate(train_dataloader):
            latents_ref = vae.encode(batch["ref"].to(vae.dtype)).latent_dist.sample()
            latents_p0 = vae.encode(batch["p0"].to(vae.dtype)).latent_dist.sample()
            latents_p1 = vae.encode(batch["p1"].to(vae.dtype)).latent_dist.sample()
            latents_ref = scale_vae_output(args.vae_type, vae, latents_ref).float()
            latents_p0 = scale_vae_output(args.vae_type, vae, latents_p0).float()
            latents_p1 = scale_vae_output(args.vae_type, vae, latents_p1).float()

            d0 = lpips(latents_ref, latents_p0)
            d1 = lpips(latents_ref, latents_p1)
            judge = batch["judge"].view(d0.shape)
            loss = rankLoss(d0, d1, judge*2-1) # convert to -1 1
            acc = compute_accuracy(d0,d1,judge)

            accelerator.backward(loss.mean())
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                for module in lpips.modules():
                    if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
                        module.weight.data = torch.clamp(module.weight.data, min=0)
                
                gathered_loss = accelerator.gather(loss.repeat(args.batch_size))
                gathered_acc = accelerator.gather(torch.tensor(acc, device=accelerator.device))
                gathered_d0, gathered_d1 = accelerator.gather(d0), accelerator.gather(d1)
                abs_diff = torch.abs(gathered_d0 - gathered_d1)
                accelerator.log({
                    "train_loss": gathered_loss.mean().item(),
                    "train_accuracy": gathered_acc.mean().item(),
                    "d0_d1_abs_diff": abs_diff.mean().item(),
                    "lr": optimizer.param_groups[0]["lr"]
                }, step=global_step)
                progress_bar.update(1)
                global_step += 1
                
        if epoch >= args.num_epochs // 2:
            old_lr = optimizer.param_groups[0]["lr"]
            next_lr = old_lr - (args.lr / (args.num_epochs // 2))
            for param_group in optimizer.param_groups:
                param_group["lr"] = next_lr
     
        accelerator.wait_for_everyone()

        # Validation
        logger.info("Running validation")
        lpips.eval()
        d0s = []
        d1s = []
        gts = []

        val_progress_bar = tqdm(total=len(valid_dataloader), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
        val_progress_bar.set_description(f"Validation Epoch: {epoch}/{args.num_epochs}")

        for step, batch in enumerate(valid_dataloader):
            latents_ref = vae.encode(batch["ref"].to(vae.dtype)).latent_dist.sample()
            latents_p0 = vae.encode(batch["p0"].to(vae.dtype)).latent_dist.sample()
            latents_p1 = vae.encode(batch["p1"].to(vae.dtype)).latent_dist.sample()
            latents_ref = scale_vae_output(args.vae_type, vae, latents_ref).float()
            latents_p0 = scale_vae_output(args.vae_type, vae, latents_p0).float()
            latents_p1 = scale_vae_output(args.vae_type, vae, latents_p1).float()
            
            with torch.no_grad():
                d0 = lpips(latents_ref, latents_p0)
                d1 = lpips(latents_ref, latents_p1)
            
            d0s.append(accelerator.gather(d0).cpu().numpy())
            d1s.append(accelerator.gather(d1).cpu().numpy())
            gts.append(accelerator.gather(batch["judge"]).cpu().numpy())
            val_progress_bar.update(1)
        
        val_progress_bar.close()

        d0s = np.concatenate(d0s).flatten()
        d1s = np.concatenate(d1s).flatten()
        gts = np.concatenate(gts).flatten()
        scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
        mean_score = np.mean(scores)

        if accelerator.is_main_process:
            accelerator.log({"valid_2afc_score": mean_score}, step=global_step)
            logger.info(f"EPOCH: {epoch}, Validation 2AFC Score: {mean_score:.4f}")
            accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-ep={epoch}"))
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    accelerator.end_training()

def eval_only(args):
    logger = get_logger(__name__, log_level="INFO")
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        even_batches=False,  # To ensure exact number of val samples
    )
    
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    vae = load_vae(args.vae_type, device=accelerator.device, dtype=torch.float16, requires_grad=False)

    lpips = LatentLPIPS(backbone_type=args.model_type, backbone_in_channels=4 if args.vae_type in ["sd1", "sdxl"] else 16,
                               pretrained_full_model_path=args.eval_model_path)

    valid_dataset = load_from_disk(args.val_dataset_name)
    with accelerator.main_process_first():
        valid_dataset = valid_dataset.with_transform(partial(preprocess, resolution=(512, 512)))
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    lpips, valid_dataloader = accelerator.prepare(lpips, valid_dataloader)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(valid_dataset)}")
    logger.info(f"  Batch size = {args.batch_size}")

    lpips.eval()
    d0s = []
    d1s = []
    gts = []
    
    progress_bar = tqdm(total=len(valid_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Evaluating")
    
    for step, batch in enumerate(valid_dataloader):
        latents_ref = vae.encode(batch["ref"].to(vae.dtype)).latent_dist.sample()
        latents_p0 = vae.encode(batch["p0"].to(vae.dtype)).latent_dist.sample()
        latents_p1 = vae.encode(batch["p1"].to(vae.dtype)).latent_dist.sample()
        latents_ref = scale_vae_output(args.vae_type, vae, latents_ref).float()
        latents_p0 = scale_vae_output(args.vae_type, vae, latents_p0).float()
        latents_p1 = scale_vae_output(args.vae_type, vae, latents_p1).float()
        
        with torch.no_grad():
            d0 = lpips(latents_ref, latents_p0)
            d1 = lpips(latents_ref, latents_p1)
        
        d0s.append(accelerator.gather(d0).cpu().numpy())
        d1s.append(accelerator.gather(d1).cpu().numpy())
        gts.append(accelerator.gather(batch["judge"]).cpu().numpy())
        
        progress_bar.update(1)

    # Gather evaluation results from all processes
    accelerator.wait_for_everyone()
    d0s = np.concatenate(d0s).flatten()
    d1s = np.concatenate(d1s).flatten()
    gts = np.concatenate(gts).flatten()

    scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
    mean_score = np.mean(scores)
    
    logger.info(f"Evaluation Results - 2AFC Score: {mean_score:.5f}, Total Samples: {len(gts)}")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        main(args)