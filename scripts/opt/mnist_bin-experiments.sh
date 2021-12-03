gpu="--gpu"
seed_array=( 1 2 3 4 5 )
digit_array=( 3 )
opt_constraint_threshold_array=( -16 -17 -19 )
query_budget=500
n_retrain_epochs=0.1
n_init_retrain_epochs=1
opt_bounds=3

k=1e-3
r=5
weight_type="rank"
lso_strategy="opt"
opt_constraint_strategy="gmm_full"
#n_gmm_components=10
model_type="vae"

for digit in "${digit_array[@]}"; do
    for c in "${opt_constraint_threshold_array[@]}"; do
        for seed in "${seed_array[@]}"; do
            # Echo info
            echo "digit=${digit} r=${r} k=${k} c=${c} seed=${seed}"

            root_dir="logs/opt/mnist_binary/${digit}"
            start_model="logs/train/mnist_binary/${digit}/vae/lightning_logs/version_0/checkpoints/last.ckpt"
            # Run the command
            python weighted_retraining/opt_scripts/opt_mnist_bin.py \
                --seed="$seed" $gpu \
                --dataset_path="data/mnist/mnist_D${digit}_Pthickness_BTrue_DF5.npz" \
                --property_key=thickness \
                --query_budget="$query_budget" \
                --retraining_frequency="$r" \
                --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/${model_type}/${opt_constraint_strategy}_${n_gmm_components}/c_${c}/seed${seed}" \
                --pretrained_model_type="$model_type" \
                --pretrained_model_file="$start_model" \
                --weight_type="$weight_type" \
                --rank_weight_k="$k" \
                --n_retrain_epochs="$n_retrain_epochs" \
                --n_init_retrain_epochs="$n_init_retrain_epochs" \
                --opt_bounds="$opt_bounds" \
                --lso_strategy="$lso_strategy" \
                --opt_constraint_threshold="$c" \
                --opt_constraint_strategy="$opt_constraint_strategy" #\
                #--n_gmm_components="$n_gmm_components"
        done
    done
done
