# Script to conduct all CelebA-VAE experiments

# Meta flags
gpu="--gpu"

# Weighted retraining hyperparameters
query_budget=500
k=1e-3
r=5
n_retrain_epochs=0.1
n_init_retrain_epochs=1
opt_bounds=3
weight_type="rank"

# Model paths
root_dir="logs/opt/celeba/vae"
start_model="logs/train/celeba/vae/lightning_logs/version_0/checkpoints/last.ckpt"
pretrained_predictor_file="logs/train/celeba-dialog-predictor/predictor_128.pth.tar"
scaled_predictor_state_dict="logs/train/celeba-dialog-predictor/predictor_128_scaled3.pth.tar"
celeba_data_path="data/celeba-dialog"

python src/opt_scripts/opt_celeba_vae.py \
    --seed="1" $gpu \
    --tensor_dir="/content/data_tensors_64" \
    --property_id=3 \
    --max_property_value=2 \
    --train_attr_path="${celeba_data_path}/train_attr_list.txt" \
    --val_attr_path="${celeba_data_path}/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="${celeba_data_path}/filename_set.pickle" \
    --attr_file="src/configs/attributes.json" \
    --query_budget="$query_budget" \
    --retraining_frequency="$r" \
    --result_root="${root_dir}/k_${k}/r_${r}/gmm_10/c_-94/seed1" \
    --pretrained_model_file="$start_model" \
    --pretrained_predictor_file="$pretrained_predictor_file" \
    --scaled_predictor_state_dict="$scaled_predictor_state_dict" \
    --weight_type="$weight_type" \
    --rank_weight_k="$k" \
    --n_retrain_epochs="$n_retrain_epochs" \
    --n_init_retrain_epochs="$n_init_retrain_epochs" \
    --n_samples="100000" \
    --n_out="$r" \
    --n_starts=10 \
    --opt_method="SLSQP" \
    --bo_surrogate="DNGO" \
    --opt_constraint_threshold="-94" \
    --opt_constraint_strategy="gmm_fit" \
    --n_gmm_components="10" \
    --mode="all" \
    --batch_size=64 \
    --sample_distribution="normal"
