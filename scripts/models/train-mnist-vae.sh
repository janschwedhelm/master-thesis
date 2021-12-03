# Script to train all MNIST-VAE models for the thesis

# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed=0
root_dir="logs/train"
digit_array=( 0 1 2 3 4 5 6 7 8 9 )

# Latent dimension = 2 and GMM-n ex-post density model
for d in "${digit_array[@]}"; do
    for c in "${threshold_array[@]}"; do
        python weighted_retraining/train_scripts/train_mnist_vae.py \
            --root_dir="$root_dir/mnist_2/${d}/vae" \
            --seed="$seed" $gpu \
            --latent_dim=2 \
            --dataset_path="data/mnist/mnist_D${d}_Pthickness_BTrue_DF5.npz" \
            --property_key=thickness \
            --max_epochs=20 \
            --beta_final=10.0 --beta_start=1e-6 \
            --beta_warmup=1000 --beta_step=1.1 --beta_step_freq=10 \
            --batch_size=16 \
            --opt_constraint_strategy="gmm_full" \
            --opt_constraint_threshold="-12"
    done
done

# Latent dimension = 2 and GMM-10 ex-post density model
for d in "${digit_array[@]}"; do
    for c in "${threshold_array[@]}"; do
        python weighted_retraining/train_scripts/train_mnist_vae.py \
            --root_dir="$root_dir/mnist_2/${d}/vae" \
            --seed="$seed" $gpu \
            --latent_dim=2 \
            --dataset_path="data/mnist/mnist_D${d}_Pthickness_BTrue_DF5.npz" \
            --property_key=thickness \
            --max_epochs=20 \
            --beta_final=10.0 --beta_start=1e-6 \
            --beta_warmup=1000 --beta_step=1.1 --beta_step_freq=10 \
            --batch_size=16 \
            --opt_constraint_strategy="gmm_fit" \
            --n_gmm_components=10 \
            --opt_constraint_threshold="-4"
    done
done

# Latent dimension = 16
for d in "${digit_array[@]}"; do
    python weighted_retraining/train_scripts/train_mnist_vae.py \
        --root_dir="$root_dir/mnist_16/${d}/vae" \
        --seed="$seed" $gpu \
        --latent_dim=16 \
        --dataset_path="data/mnist/mnist_D${d}_Pthickness_BTrue_DF5.npz" \
        --property_key=thickness \
        --max_epochs=20 \
        --beta_final=10.0 --beta_start=1e-6 \
        --beta_warmup=1000 --beta_step=1.1 --beta_step_freq=10 \
        --batch_size=16 \
        --opt_constraint_strategy="gmm_fit" \
        --n_gmm_components=10 \
        --opt_constraint_threshold="-24.5"
done
