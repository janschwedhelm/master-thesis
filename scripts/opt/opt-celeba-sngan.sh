# Script to conduct all CelebA-SNGAN experiments

# Meta flags
gpu="--gpu"

# Weighted retraining hyperparameters
query_budget=500
k=1e-3
r=50
n_retrain_steps=222
n_init_retrain_steps=2222
opt_bounds=3
weight_type="rank"

# Model paths
pretrained_predictor_file="logs/train/celeba-dialog-predictor/predictor_128.pth.tar"
scaled_predictor_state_dict="logs/train/celeba-dialog-predictor/predictor_128_scaled3.pth.tar"
celeba_data_path="data/celeba-dialog"
sample_distribution="normal"
pretrained_model_prior="normal"


# Optimization approach
root_dir="logs/opt/celeba/sn-gan/optimization"
start_model_netg="logs/train/celeba/sn-gan/z_64/checkpoints/netG/netG_80000_steps.pth"
start_model_netd="logs/train/celeba/sn-gan/z_64/checkpoints/netD/netD_80000_steps.pth"
python src/opt_scripts/opt_celeba_sngan.py \
    --seed="1" \
    --tensor_dir="/content/data_tensors_64" \
    --property_id=3 \
    --max_property_value=2 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --attr_file="src/configs/attributes.json" \
    --query_budget="$query_budget" \
    --retraining_frequency="$r" \
    --result_root="${root_dir}/k_${k}/r_${r}/seed1" \
    --pretrained_netg_model_file="$start_model_netg" \
    --pretrained_netd_model_file="$start_model_netd" \
    --pretrained_model_prior="$pretrained_model_prior" \
    --pretrained_predictor_file="$pretrained_predictor_file" \
    --scaled_predictor_state_dict="$scaled_predictor_state_dict" \
    --weight_type="$weight_type" \
    --rank_weight_k="$k" \
    --n_retrain_steps="$n_retrain_steps" \
    --n_init_retrain_steps="$n_init_retrain_steps" \
    --sample_distribution="$sample_distribution" \
    --opt_method="SLSQP" \
    --bo_surrogate="DNGO" \
    --mode="all" \
    --batch_size=64 \
    --n_samples=100000 \
    --n_best_points=100000 \
    --n_rand_points=0 \
    --opt_constraint_strategy="discriminator" \
    --opt_constraint_threshold=0.5


# Sampling approach
root_dir="logs/opt/celeba/sn-gan/sampling"
start_model_netg="logs/train/celeba/sn-gan/z_128/checkpoints/netG/netG_80000_steps.pth"
start_model_netd="logs/train/celeba/sn-gan/z_128/checkpoints/netD/netD_80000_steps.pth"
python src/opt_scripts/opt_celeba_sngan.py \
    --seed="1" \
    --tensor_dir="/content/data_tensors_64" \
    --property_id=3 \
    --max_property_value=2 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --attr_file="src/configs/attributes.json" \
    --query_budget="$query_budget" \
    --retraining_frequency="$r" \
    --result_root="${root_dir}/k_${k}/r_${r}/seed1" \
    --pretrained_netg_model_file="$start_model_netg" \
    --pretrained_netd_model_file="$start_model_netd" \
    --pretrained_model_prior="$pretrained_model_prior" \
    --pretrained_predictor_file="$pretrained_predictor_file" \
    --scaled_predictor_state_dict="$scaled_predictor_state_dict" \
    --weight_type="$weight_type" \
    --rank_weight_k="$k" \
    --n_retrain_steps="$n_retrain_steps" \
    --n_init_retrain_steps="$n_init_retrain_steps" \
    --sample_distribution="$sample_distribution" \
    --opt_method="sampling" \
    --bo_surrogate="DNGO" \
    --mode="all" \
    --batch_size=64 \
    --n_samples=100000 \
    --n_best_points=100000 \
    --n_rand_points=0 \
    --opt_constraint_strategy="discriminator" \
    --opt_constraint_threshold=0.5
