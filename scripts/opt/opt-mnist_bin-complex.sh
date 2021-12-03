# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed_array=( 1 )
digit_array=( 3 )
type_array=( vae )
bo_surrogate_array=( DNGO )
opt_method_array=( SLSQP )
root_dir="logs/opt/mnist_binary_complex"
start_model="logs/train/mnist_binary_complex/3/vae/lightning_logs/version_0/checkpoints/last.ckpt"
query_budget=500
n_retrain_epochs=0.1
n_init_retrain_epochs=1
opt_constraint_strategy="gmm_fit"
n_samples=100000
sample_distribution="uniform"


k=1e-3
r=5
c=-24.5
weight_type="rank"
lso_strategy="opt"
opt_method="SLSQP"
bo_surrogate="DNGO"
n_gmm_components=10


for t in "${type_array[@]}"; do
    for digit in "${digit_array[@]}"; do
        for seed in "${seed_array[@]}"; do
            for bo_surrogate in "${bo_surrogate_array[@]}"; do
                for opt_method in "${opt_method_array[@]}"; do
                    # Echo info
                    echo "digit=${digit} seed=${seed} c=${c} r=${r} k=${k} opt_method=${opt_method} bo_surrogate=${bo_surrogate}"
        
                    root_dir="logs/opt/mnist_binary_complex_1fit_16/sparse_out/${digit}/${t}"
                    start_model="logs/train/mnist_binary_complex/${digit}/${t}/lightning_logs/version_0/checkpoints/last.ckpt"
                    # Run the command
                    python weighted_retraining/opt_scripts/opt_mnist_bin_complex.py \
                        --seed="$seed" $gpu \
                        --dataset_path=data/mnist/mnist_D${digit}_Pthickness_BTrue_DF5.npz \
                        --property_key=thickness \
                        --query_budget="$query_budget" \
                        --retraining_frequency="$r" \
                        --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/${opt_constraint_strategy}_${n_gmm_components}/c_${c}/seed${seed}_test_${bo_surrogate}" \
                        --pretrained_model_file="$start_model" \
                        --weight_type="$weight_type" \
                        --rank_weight_k="$k" \
                        --n_retrain_epochs="$n_retrain_epochs" \
                        --n_init_retrain_epochs="$n_init_retrain_epochs" \
                        --lso_strategy="$lso_strategy" \
                        --n_samples="$n_samples" \
                        --sample_distribution="$sample_distribution" \
                        --pretrained_model_type="$t" \
                        --n_out="$r" \
                        --n_starts=10 \
                        --opt_method="$opt_method" \
                        --bo_surrogate="$bo_surrogate" \
                        --opt_constraint_threshold="$c" \
                        --opt_constraint_strategy="$opt_constraint_strategy" \
                        --n_gmm_components="$n_gmm_components"
                done
            done
        done
    done
done