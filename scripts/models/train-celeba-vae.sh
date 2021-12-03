# Script to train CelebA-VAE model for the thesis

# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed=0
root_dir="logs/train"
celeba_data_path="data/celeba-dialog"

python weighted_retraining/train_scripts/train_celeba_vae.py \
    --root_dir="$root_dir/celeba/vae" \
    --seed="$seed" $gpu \
    --latent_dim=64 \
    --tensor_dir="/content/data_tensors_64" \
    --property_id=3 \
    --max_property_value=2 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --max_epochs=30 \
    --beta_final=0.21471 --beta_start=1e-6 \
    --beta_warmup=5000 --beta_step=1.1 --beta_step_freq=50 \
    --batch_size=64 \
    --mode="all" \
    --lr=0.0001
    