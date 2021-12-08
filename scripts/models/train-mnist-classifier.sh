# Script to train MNIST classifier model for the thesis

# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed=0
root_dir="logs/train"

python src/train_scripts/train_mnist_classifier.py \
    --root_dir="$root_dir/mnist_classifier" \
    --seed="$seed" $gpu \
    --dataset_path=data/mnist/mnist_BTrue.npz \
    --max_epochs=20 \
    --batch_size=32
