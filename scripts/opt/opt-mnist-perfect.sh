# Script to conduct all MNIST experiments using 'perfect' optimizer strategy for the thesis

# Meta flags
gpu="--gpu"

seed_array=( 1 2 3 4 5 )
digit_array=( 3 )

# Weighted retraining hyperparameters
query_budget=500
k=1e-3
r=5
n_retrain_epochs=0.1
n_init_retrain_epochs=1
opt_bounds=3
weight_type="rank"

# VAE experiments with GMM-N constraint
for digit in "${digit_array[@]}"; do
    for seed in "${seed_array[@]}"; do
        root_dir="logs/opt/mnist_z2/${digit}/vae"
        start_model="logs/train/mnist_z2/${digit}/vae/lightning_logs/version_0/checkpoints/last.ckpt"
        python src/opt_scripts/opt_mnist_perfect.py \
            --seed="$seed" $gpu \
            --dataset_path="data/mnist/mnist_D${digit}_Pthickness_BTrue_DF5.npz" \
            --property_key=thickness \
            --query_budget="$query_budget" \
            --retraining_frequency="$r" \
            --result_root="${root_dir}/k_${k}/r_${r}/gmm_n/c_-12/seed${seed}" \
            --pretrained_model_type="vae" \
            --pretrained_model_file="$start_model" \
            --weight_type="$weight_type" \
            --rank_weight_k="$k" \
            --n_retrain_epochs="$n_retrain_epochs" \
            --n_init_retrain_epochs="$n_init_retrain_epochs" \
            --opt_bounds="$opt_bounds" \
            --opt_constraint_strategy="gmm_full" \
            --opt_constraint_threshold="-12"
    done
done

# VAE experiments with GMM-10 constraint
for digit in "${digit_array[@]}"; do
    for seed in "${seed_array[@]}"; do
        root_dir="logs/opt/mnist_z2/${digit}/vae"
        start_model="logs/train/mnist_z2/${digit}/vae/lightning_logs/version_0/checkpoints/last.ckpt"
        python src/opt_scripts/opt_mnist_perfect.py \
            --seed="$seed" $gpu \
            --dataset_path="data/mnist/mnist_D${digit}_Pthickness_BTrue_DF5.npz" \
            --property_key=thickness \
            --query_budget="$query_budget" \
            --retraining_frequency="$r" \
            --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/gmm_10/c_-4/seed${seed}" \
            --pretrained_model_type="vae" \
            --pretrained_model_file="$start_model" \
            --weight_type="$weight_type" \
            --rank_weight_k="$k" \
            --n_retrain_epochs="$n_retrain_epochs" \
            --n_init_retrain_epochs="$n_init_retrain_epochs" \
            --opt_bounds="$opt_bounds" \
            --opt_constraint_strategy="gmm_fit" \
            --opt_constraint_threshold="-4" \
            --n_gmm_components=10

    done
done

# RAE experiments with GMM-10 constraint
for digit in "${digit_array[@]}"; do
    for seed in "${seed_array[@]}"; do
        root_dir="logs/opt/mnist_z2/${digit}/rae"
        start_model="logs/train/mnist_z2/${digit}/rae/lightning_logs/version_0/checkpoints/last.ckpt"
        python src/opt_scripts/opt_mnist_perfect.py \
            --seed="$seed" $gpu \
            --dataset_path="data/mnist/mnist_D${digit}_Pthickness_BTrue_DF5.npz" \
            --property_key=thickness \
            --query_budget="$query_budget" \
            --retraining_frequency="$r" \
            --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/gmm_10/c_-4.5/seed${seed}" \
            --pretrained_model_type="rae" \
            --pretrained_model_file="$start_model" \
            --weight_type="$weight_type" \
            --rank_weight_k="$k" \
            --n_retrain_epochs="$n_retrain_epochs" \
            --n_init_retrain_epochs="$n_init_retrain_epochs" \
            --opt_bounds="$opt_bounds" \
            --opt_constraint_strategy="gmm_fit" \
            --opt_constraint_threshold="-4.5" \
            --n_gmm_components=10
    done
done
