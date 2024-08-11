import os
import argparse
import math
import logging
import numpy as np
from functools import partial
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision
import diffusers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from datasets import load_from_disk
from safetensors.torch import load_file
from elatentlpips import VGG16_Latent, VGG16_Latent_BN, VGG16_Latent_GN
from utils import load_vae, scale_vae_output

def parse_args():
    parser = argparse.ArgumentParser(description="Train latent vgg model")
    parser.add_argument("--eval_only", action="store_true", help="Whether to only evaluate the model")
    parser.add_argument("--eval_model_path", type=str, default="checkpoints/latent_vgg16_gn_sd1/checkpoint-ep=99/model.safetensors", help="path to the model checkpoint for evaluation")

    parser.add_argument("--dataset_name", type=str, default="/root/data/dataset/vision_general/ImageNet/imagenet_dataset_train", help="path to the training dataset")
    parser.add_argument("--val_dataset_name", type=str, default="/root/data/dataset/vision_general/ImageNet/imagenet_dataset_val", help="path to the validation dataset")
    parser.add_argument("--vae_type", type=str, default="sd1", choices=["sd1", "sdxl", "sd3", "flux"], help="type of VAE to use")

    parser.add_argument("--model_type", type=str, default="VGG16_Latent_GN", choices=["VGG16_Latent", "VGG16_Latent_BN", "VGG16_Latent_GN"], help="model type to use")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training (effective batch size will be batch_size * gradient_accumulation_steps * num_processes")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for optimizer")

    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="number of workers for the dataloader")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision, requires specific hardware and software")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint folder to resume from (e.g. checkpoint-1000)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/latent_vgg16_gn_sd1", help="Directory for model predictions and checkpoints")
    parser.add_argument("--seed", type=int, default=4727, help="Random seed for reproducibility")

    args = parser.parse_args()
    return args

def preprocess(examples, transforms):
    images = [np.array(transforms(Image.open(img).convert("RGB"))).transpose(2,0,1) for img in examples["image"]]
    examples["image_tensors"] = 2 * (torch.tensor(np.stack(images)) / 255) - 1
    examples["label_tensors"] = torch.tensor(examples["label"])
    return examples

def collate_fn(examples):
    images = torch.stack([example["image_tensors"] for example in examples])
    labels = torch.stack([example["label_tensors"] for example in examples])
    return {"image": images, "label": labels}

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main(args):
    logger = get_logger(__name__, log_level="INFO")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
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

    if args.model_type == "VGG16_Latent":
        model = VGG16_Latent(in_channels=4 if args.vae_type in ["sd1", "sdxl"] else 16)
    elif args.model_type == "VGG16_Latent_BN":
        model = VGG16_Latent_BN(in_channels=4 if args.vae_type in ["sd1", "sdxl"] else 16)
    else:
        model = VGG16_Latent_GN(in_channels=4 if args.vae_type in ["sd1", "sdxl"] else 16)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_dataset = load_from_disk(args.dataset_name)
    valid_dataset = load_from_disk(args.val_dataset_name)
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(512),
        torchvision.transforms.RandomHorizontalFlip()
    ])
    valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(600),
        torchvision.transforms.CenterCrop(512)
    ])
    with accelerator.main_process_first():
        train_dataset = train_dataset.with_transform(partial(preprocess, transforms=train_transforms))
        valid_dataset = valid_dataset.with_transform(partial(preprocess, transforms=valid_transforms))
    
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
    )

    model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("E-Latent LPIPS", config=vars(args))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Num Update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {num_update_steps_per_epoch * args.num_epochs}")
    logger.info(f"  Total trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    first_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint is not None:
        first_epoch = int(args.resume_from_checkpoint.split("ep=")[-1]) + 1
        global_step = first_epoch * num_update_steps_per_epoch
        accelerator.load_state(os.path.join(args.output_dir, args.resume_from_checkpoint))

    for epoch in range(first_epoch, args.num_epochs):
        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
        progress_bar.set_description(f"Epoch: {epoch}/{args.num_epochs}")
        log_train_loss = 0.0
        log_train_accuracy = 0.0
        num_train_samples = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                latents = vae.encode(batch["image"].to(vae.dtype)).latent_dist.sample()
                latents = scale_vae_output(args.vae_type, vae, latents).float()
                
                output = model(latents)
                loss = F.cross_entropy(output, batch["label"])
                acc1, = accuracy(output, batch["label"])

                accelerator.backward(loss)
                optimizer.step()
                
                log_train_loss += loss.item() * batch["label"].size(0)
                log_train_accuracy += acc1.item() * batch["label"].size(0)
                num_train_samples += batch["label"].size(0)
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                # Gather metrics from all processes
                all_train_loss = accelerator.gather(torch.tensor(log_train_loss, device=accelerator.device)).sum().item()
                all_train_accuracy = accelerator.gather(torch.tensor(log_train_accuracy, device=accelerator.device)).sum().item()
                all_train_samples = accelerator.gather(torch.tensor(num_train_samples, device=accelerator.device)).sum().item()

                # Calculate average metrics
                avg_train_loss = all_train_loss / all_train_samples
                avg_train_accuracy = all_train_accuracy / all_train_samples

                accelerator.log({
                    "train_loss": avg_train_loss,
                    "train_accuracy": avg_train_accuracy,
                    "lr": scheduler.get_last_lr()[0]
                }, step=global_step)
                
                log_train_loss = 0.0
                log_train_accuracy = 0.0
                num_train_samples = 0
                progress_bar.update(1)
                global_step += 1
        
        scheduler.step()
        accelerator.wait_for_everyone()
        # Validation
        logger.info("Running validation")
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        total_samples = 0
        for step, batch in enumerate(valid_dataloader):
            latents = vae.encode(batch["image"].to(vae.dtype)).latent_dist.sample()
            latents = scale_vae_output(args.vae_type, vae, latents).float()
            
            with torch.no_grad():
                output = model(latents)
                loss = F.cross_entropy(output, batch["label"])
                
                valid_loss += loss.item() * batch["label"].size(0)
                _, predicted = output.max(1)
                valid_correct += predicted.eq(batch["label"]).sum().item()
                total_samples += batch["label"].size(0)
        
        # Gather validation results from all processes
        accelerator.wait_for_everyone()
        valid_loss = accelerator.gather(torch.tensor(valid_loss, device=accelerator.device)).sum().item()
        valid_correct = accelerator.gather(torch.tensor(valid_correct, device=accelerator.device)).sum().item()
        total_samples = accelerator.gather(torch.tensor(total_samples, device=accelerator.device)).sum().item()

        valid_loss /= total_samples
        valid_accuracy = 100.0 * valid_correct / total_samples
        
        if accelerator.is_main_process:
            accelerator.log({
                "valid_loss": valid_loss,
                "valid_accuracy": valid_accuracy
            }, step=global_step)
            
            logger.info(f"EPOCH: {epoch}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")
            accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-ep={epoch}"))
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    accelerator.end_training()

def eval_only(args):
    logger = get_logger(__name__, log_level="INFO")
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        even_batches=False, # To ensure exact 50k val samples
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

    if args.model_type == "VGG16_Latent":
        model = VGG16_Latent(in_channels=4 if args.vae_type in ["sd1", "sdxl"] else 16)
    elif args.model_type == "VGG16_Latent_BN":
        model = VGG16_Latent_BN(in_channels=4 if args.vae_type in ["sd1", "sdxl"] else 16)
    else:
        model = VGG16_Latent_GN(in_channels=4 if args.vae_type in ["sd1", "sdxl"] else 16)
    
    if args.eval_model_path.endswith(".safetensors"):
        model.load_state_dict(load_file(args.eval_model_path))
    else:
        model.load_state_dict(torch.load(args.eval_model_path, map_location='cpu'))

    valid_dataset = load_from_disk(args.val_dataset_name)
    valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(600),
        torchvision.transforms.CenterCrop(512)
    ])
    with accelerator.main_process_first():
        valid_dataset = valid_dataset.with_transform(partial(preprocess, transforms=valid_transforms))
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    model, valid_dataloader = accelerator.prepare(model, valid_dataloader)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(valid_dataset)}")
    logger.info(f"  Batch size = {args.batch_size}")

    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(total=len(valid_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Evaluating")
    
    for step, batch in enumerate(valid_dataloader):
        latents = vae.encode(batch["image"].to(vae.dtype)).latent_dist.sample()
        latents = scale_vae_output(args.vae_type, vae, latents).float()
        
        with torch.no_grad():
            output = model(latents.float())
            loss = F.cross_entropy(output, batch["label"])
            
            valid_loss += loss.item() * batch["label"].size(0)
            _, predicted = output.max(1)
            valid_correct += predicted.eq(batch["label"]).sum().item()
            total_samples += batch["label"].size(0)
        
        progress_bar.update(1)

    # Gather validation results from all processes
    accelerator.wait_for_everyone()
    valid_loss = accelerator.gather(torch.tensor(valid_loss, device=accelerator.device)).sum().item()
    valid_correct = accelerator.gather(torch.tensor(valid_correct, device=accelerator.device)).sum().item()
    total_samples = accelerator.gather(torch.tensor(total_samples, device=accelerator.device)).sum().item()

    valid_loss /= total_samples
    valid_accuracy = 100.0 * valid_correct / total_samples
    
    logger.info(f"Evaluation Results - Loss: {valid_loss:.5f}, Accuracy: {valid_accuracy:.5f}%, Total Samples: {total_samples}")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        main(args)