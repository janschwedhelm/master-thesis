# Script to conduct all MNIST experiments using traditional local optimization strategies for the thesis

# Meta flags
gpu="--gpu"

seed_array=( 1 2 3 4 5 )

# Weighted retraining hyperparameters
query_budget=500
k=1e-3
r=5
n_retrain_epochs=0.1
n_init_retrain_epochs=1
opt_bounds=3
weight_type="rank"
bo_surrogate_array=( GP DNGO )
root_dir="logs/opt/mnist_z16"
start_model="logs/train/mnist_z16/3/vae/lightning_logs/version_0/checkpoints/last.ckpt"

for seed in "${seed_array[@]}"; do
    for bo_surrogate in "${bo_surrogate_array[@]}"; do
            root_dir="logs/opt/mnist_z16/3/vae"
            start_model="logs/train/mnist_z16/3/vae/lightning_logs/version_0/checkpoints/last.ckpt"
            python src/opt_scripts/opt_mnist_bin_complex.py \
                --seed="$seed" $gpu \
                --dataset_path=data/mnist/mnist_D${digit}_Pthickness_BTrue_DF5.npz \
                --property_key=thickness \
                --query_budget="$query_budget" \
                --retraining_frequency="$r" \
                --result_root="${root_dir}/k_${k}/r_${r}/gmm_10/c_-24.5/seed${seed}" \
                --pretrained_model_file="$start_model" \
                --weight_type="$weight_type" \
                --rank_weight_k="$k" \
                --n_retrain_epochs="$n_retrain_epochs" \
                --n_init_retrain_epochs="$n_init_retrain_epochs" \
                --n_samples="100000" \
                --sample_distribution="uniform" \
                --pretrained_model_type="vae" \
                --n_out="$r" \
                --n_starts=10 \
                --opt_method="SLSQP" \
                --bo_surrogate="$bo_surrogate" \
                --opt_constraint_threshold="-24.5" \
                --opt_constraint_strategy="gmm_fit" \
                --n_gmm_components="10"
    done
done
