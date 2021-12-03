# Script to train all MNIST-RAE models for the thesis

# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed=0
root_dir="logs/train"
digit_array=( 0 1 2 3 4 5 6 7 8 9 )

for d in "${digit_array[@]}"; do
    python weighted_retraining/train_scripts/train_mnist_rae.py \
        --root_dir="$root_dir/mnist_2/${d}/rae" \
        --seed="$seed" $gpu \
        --latent_dim=2 \
        --dataset_path=data/mnist/mnist_D${d}_Pthickness_BTrue_DF5.npz \
        --property_key=thickness \
        --max_epochs=20 \
        --batch_size=16 \
        --latent_emb_weight=1 \
        --reg_weight=1 \
        --opt_constraint_strategy="gmm_fit" \
        --n_gmm_components=10 \
        --opt_constraint_threshold="-4.5"
done
    