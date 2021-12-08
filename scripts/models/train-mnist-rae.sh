# Script to train all MNIST-RAE models for the thesis

# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed=0
root_dir="logs/train"
digit_array=( 3 )

for d in "${digit_array[@]}"; do
    python src/train_scripts/train_mnist_rae.py \
        --root_dir="$root_dir/mnist_z2/${d}/rae" \
        --seed="$seed" $gpu \
        --latent_dim=2 \
        --dataset_path=data/mnist/mnist_D${d}_Pthickness_BTrue_DF5.npz \
        --property_key=thickness \
        --max_epochs=20 \
        --batch_size=16 \
        --latent_emb_weight=1 \
        --reg_weight=1
done
    
