# Script to train CelebA-VQ-VAE model for the thesis

# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed=0
root_dir="logs/train"
celeba_data_path="data/celeba-dialog"

# Train shapes VAE
  python weighted_retraining/train_scripts/train_celeba_vqvae.py \
      --root_dir="$root_dir/celeba/vq-vae" \
      --seed="$seed" $gpu \
      --num_embeddings=256 \
      --embedding_dim=64 \
      --tensor_dir="/content/data_tensors_64" \
      --property_id=3 \
      --max_property_value=2 \
      --train_attr_path="$celeba_data_path/train_attr_list.txt" \
      --val_attr_path="$celeba_data_path/val_attr_list.txt" \
      --filename_set_path="$celeba_data_path/filename_set.pickle" \
      --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
      --max_epochs=30 \
      --beta=0.25 \
      --lr=0.001 \
      --mode="all"
