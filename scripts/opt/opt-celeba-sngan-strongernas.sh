# Script to conduct all CelebA-SNGAN experiments

# Meta flags
gpu=""

# Weighted retraining hyperparameters
query_budget=10000
k=1e-3
r=100
n_retrain_steps=1
n_init_retrain_steps=1
opt_bounds=3
weight_type="rank"

# Model paths
pretrained_predictor_file="logs/train/celeba-dialog-predictor/predictor_128.pth.tar"
scaled_predictor_state_dict="logs/train/celeba-dialog-predictor/predictor_128_scaled3.pth.tar"
celeba_data_path="data/celeba-dialog"
sample_distribution="normal"
pretrained_model_prior="normal"


# Sampling approach
root_dir="logs/opt/celeba/sn-gan/sampling_strongernas"
start_model_netg="logs/train/celeba/sn-gan/z_128/checkpoints/netG/netG_80000_steps.pth"
start_model_netd="logs/train/celeba/sn-gan/z_128/checkpoints/netD/netD_80000_steps.pth"

python src/opt_scripts/opt_celeba_sngan_strongernas.py \
    --seed="1" \
    --tensor_dir="data/celeba-dialog/data_tensors_64" \
    --property_id=3 \
    --max_property_value=2 \
    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
    --filename_set_path="$celeba_data_path/filename_set.pickle" \
    --attr_file="src/configs/attributes.json" \
    --query_budget="$query_budget" \
    --retraining_frequency="$r" \
    --result_root="${root_dir}/k_${k}/r_${r}/seed1_1000_2" \
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
    --bo_surrogate="GP" \
    --mode="all" \
    --batch_size=64 \
    --M_0=1000 \
    --n_best_points=1000 \
    --n_rand_points=0 \
    --M=10 \
    --N=1000 \
    --opt_constraint_strategy="discriminator" \
    --opt_constraint_threshold=0.5 \
    --n_inducing_points=500
