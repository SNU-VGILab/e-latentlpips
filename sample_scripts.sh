# Sample reference scripts for running the code.
# Change the paths and configurations as per your requirements and uncomment the commands to run the code. (bash sample_scripts.sh)
# Note that I've moved checkpoints to /root/data2/e-latentlpips/ after training...!

# Commands for dataset_preparation, for this part, make sure to run inside the dataset_preparation directory
# We assume that the data is already downloaded
# python create_imagenet_dataset.py --data_root /root/data/dataset/vision_general/ImageNet --phase train --save_path /root/data/dataset/vision_general/ImageNet/imagenet_dataset_train
# python create_imagenet_dataset.py --data_root /root/data/dataset/vision_general/ImageNet --phase val --save_path /root/data/dataset/vision_general/ImageNet/imagenet_dataset_val
# python create_2afc_dataset.py --data_root /root/data/dataset/vision_general/bapps/2afc --phase train --save_path /root/data/dataset/vision_general/bapps/2afc/2afc_dataset_train
# python create_2afc_dataset.py --data_root /root/data/dataset/vision_general/bapps/2afc --phase val --save_path /root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val
# python create_2afc_dataset.py --data_root /root/data/dataset/vision_general/bapps/2afc --phase val_traditional --save_path /root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val_traditional
# python create_2afc_dataset.py --data_root /root/data/dataset/vision_general/bapps/2afc --phase val_cnn --save_path /root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val_cnn
# python create_2afc_dataset.py --data_root /root/data/dataset/vision_general/bapps/2afc --phase val_real --save_path /root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val_real


# Commands for training_latent_vgg.py
# export DATASET_ID_TRAIN="/root/data/dataset/vision_general/ImageNet/imagenet_dataset_train"
# export DATASET_ID_VAL="/root/data/dataset/vision_general/ImageNet/imagenet_dataset_val"
# export OUTPUT_DIR="checkpoints/latent_vgg16_gn_sd1"
# export ACCELERATE_CONFIG_FILE="accelerate_configs/8gpu.yaml"
# accelerate launch --config_file=$ACCELERATE_CONFIG_FILE train_latent_vgg.py \
#     --dataset_name $DATASET_ID_TRAIN --val_dataset_name $DATASET_ID_VAL --output_dir $OUTPUT_DIR \
#     --batch_size 64 --vae_type "sd1" --model_type "VGG16_Latent_GN" --lr 0.1

# Commands for evaluating the trained latent_vgg model
# accelerate launch --config_file ./accelerate_configs/1gpu.yaml train_latent_vgg.py --eval_only \
#     --eval_model_path "/root/data2/e-latentlpips/latent_vgg16_gn_sd1/checkpoint-ep=98/model.safetensors" \
#     --batch_size 64 --vae_type "sd1" --model_type "VGG16_Latent_GN"


# Commands for training_latent_lpips.py
# export DATASET_ID_TRAIN="/root/data/dataset/vision_general/bapps/2afc/2afc_dataset_train"
# export DATASET_ID_VAL="/root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val"
# export OUTPUT_DIR="checkpoints/latent_lpips_vgg16_gn_sd1"
# export ACCELERATE_CONFIG_FILE="accelerate_configs/5gpu.yaml"
# accelerate launch --config_file=$ACCELERATE_CONFIG_FILE train_latent_lpips.py \
#     --dataset_name $DATASET_ID_TRAIN --val_dataset_name $DATASET_ID_VAL --output_dir $OUTPUT_DIR \
#     --batch_size 10 --vae_type "sd1" --model_type "VGG16_Latent_GN" \
#     --pretrained_backbone_path "/root/data2/e-latentlpips/latent_vgg16_gn_sd1/checkpoint-ep=98/model.safetensors"

# Commands for evaluating the trained latent_lpips model
# accelerate launch --config_file ./accelerate_configs/1gpu.yaml train_latent_lpips.py --eval_only \
#     --eval_model_path "/root/data2/e-latentlpips/latent_lpips_vgg16_gn_sd1/checkpoint-ep=0/model.safetensors" \
#     --batch_size 64 --vae_type "sd1" --model_type "VGG16_Latent_GN" --val_dataset_name /root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val
# accelerate launch --config_file ./accelerate_configs/1gpu.yaml train_latent_lpips.py --eval_only \
#     --eval_model_path "/root/data2/e-latentlpips/latent_lpips_vgg16_gn_sd1/checkpoint-ep=0/model.safetensors" \
#     --batch_size 64 --vae_type "sd1" --model_type "VGG16_Latent_GN" --val_dataset_name /root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val_traditional
# accelerate launch --config_file ./accelerate_configs/1gpu.yaml train_latent_lpips.py --eval_only \
#     --eval_model_path "/root/data2/e-latentlpips/latent_lpips_vgg16_gn_sd1/checkpoint-ep=0/model.safetensors" \
#     --batch_size 64 --vae_type "sd1" --model_type "VGG16_Latent_GN" --val_dataset_name /root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val_cnn
# accelerate launch --config_file ./accelerate_configs/1gpu.yaml train_latent_lpips.py --eval_only \
#     --eval_model_path "/root/data2/e-latentlpips/latent_lpips_vgg16_gn_sd1/checkpoint-ep=0/model.safetensors" \
#     --batch_size 64 --vae_type "sd1" --model_type "VGG16_Latent_GN" --val_dataset_name /root/data/dataset/vision_general/bapps/2afc/2afc_dataset_val_real


# Commands for train_overfit.py
# python train_overfit.py --output_dir "outputs/sd1_bg+cutout" --vae_type "sd1" --aug_type "bg+cutout" \
#     --lpips_domain "latent" --pretrained_latent_lpips_path "/root/data2/e-latentlpips/latent_lpips_vgg16_gn_sd1/checkpoint-ep=0/model.safetensors" \
#     --cuda_id 0 --lr 1e-3
# python train_overfit.py --output_dir "outputs/pixelLPIPS" --vae_type "sd1" \
#     --lpips_domain "pixel" --cuda_id 0 --lr 1e-3
