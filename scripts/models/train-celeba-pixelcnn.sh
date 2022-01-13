# Script to train CelebA-VQVAE model for the thesis

# Meta flags
gpu=""  # change to "" if no GPU is to be used
seed=0
root_dir="logs/train"
celeba_data_path="data/celeba-dialog"

# Train shapes VAE
  python src/train_scripts/train_celeba_vqvae.py \
      --root_dir="$root_dir/celeba/pixelcnn" \
      --seed="$seed" $gpu \
      --input_dim=256 \
      --dim=64 \
      --tensor_dir="/content/data_tensors_64" \
      --property_id=3 \
      --max_property_value=5 \
      --min_property_value=0 \
      --train_attr_path="$celeba_data_path/train_attr_list.txt" \
      --val_attr_path="$celeba_data_path/val_attr_list.txt" \
      --filename_set_path="$celeba_data_path/filename_set.pickle" \
      --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
      --max_epochs=100 \
      --lr=3e-4 \
      --mode="all" \
      --batch_size=128 \
      --n_layers=15 \
      --vq_vae="$root_dir/celeba/vq-vae/lightning_logs/version_0/checkpoints/last.ckpt"
