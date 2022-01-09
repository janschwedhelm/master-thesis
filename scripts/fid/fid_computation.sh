# Script to train CelebA-SNGAN model for the thesis

# Meta flags
gpu=""  # change to "" if no GPU is to be used
seed=0
celeba_data_path="data/celeba-dialog"

python src/fid/calculate_fid_scores.py \
    --seed="1" $gpu \
    --tensor_dir="$celeba_data_path/data_tensors_64" \
    --property_id=3 \
    --max_property_value=5 \
    --min_property_value=3 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --batch_size=64 \
    --mode="all" \
    --sample_path="logs/opt/celeba/vq-vae/k_1e-3/r_5/seed1/results.npz" \
    --result_dir="logs/fid/vq-vae.pkl"

python src/fid/calculate_fid_scores.py \
    --seed="1" $gpu \
    --tensor_dir="$celeba_data_path/data_tensors_64" \
    --property_id=3 \
    --max_property_value=5 \
    --min_property_value=3 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --batch_size=64 \
    --mode="all" \
    --sample_path="logs/opt/celeba/sn-gan/sampling/k_1e-3/r_50/seed1/results.npz" \
    --result_dir="logs/fid/sn-gan_sampling.pkl"

python src/fid/calculate_fid_scores.py \
    --seed="1" $gpu \
    --tensor_dir="$celeba_data_path/data_tensors_64" \
    --property_id=3 \
    --max_property_value=5 \
    --min_property_value=3 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --batch_size=64 \
    --mode="all" \
    --sample_path="logs/opt/celeba/sn-gan/optimization/k_1e-3/r_50/seed1/results.npz" \
    --result_dir="logs/fid/sn-gan_optimization.pkl"

python src/fid/calculate_fid_scores.py \
    --seed="1" $gpu \
    --tensor_dir="$celeba_data_path/data_tensors_64" \
    --property_id=3 \
    --max_property_value=5 \
    --min_property_value=3 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --batch_size=64 \
    --mode="all" \
    --sample_path="logs/opt/celeba/vae/k_1e-3/r_5/gmm_10/c_-94/seed1/results.npz" \
    --result_dir="logs/fid/vae.pkl"


