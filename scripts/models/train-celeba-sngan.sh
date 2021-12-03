# Script to train CelebA-SNGAN model for the thesis

# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed=0
root_dir="logs/train"
celeba_data_path="data/celeba-dialog"

# Latent dimension = 64
python weighted_retraining/train_scripts/train_celeba_sngan.py \
    --root_dir="$root_dir/celeba/sn-gan/z_64" \
    --seed="$seed" \
    --latent_dim=128 \
    --tensor_dir="/content/data_tensors_64" \
    --property_id=3 \
    --max_property_value=2 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --batch_size=64 \
    --mode="all"

# Latent dimension = 128
python weighted_retraining/train_scripts/train_celeba_sngan.py \
    --root_dir="$root_dir/celeba/sn-gan/z_128" \
    --seed="$seed" \
    --latent_dim=128 \
    --tensor_dir="/content/data_tensors_64" \
    --property_id=3 \
    --max_property_value=2 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --batch_size=64 \
    --mode="all"
    