gpu="--gpu"
seed_array=( 1 )
type_array=( vae )
bo_surrogate_array=( DNGO )
opt_method_array=( SLSQP )
query_budget=500
n_retrain_epochs=0.1
n_init_retrain_epochs=1
opt_constraint_strategy="gmm_fit"
c=-94
n_gmm_components=10
n_samples=100000
sample_distribution="normal"


k=1e-3
r=5
weight_type="rank"
lso_strategy="opt"

for t in "${type_array[@]}"; do
    for seed in "${seed_array[@]}"; do
        for bo_surrogate in "${bo_surrogate_array[@]}"; do
            for opt_method in "${opt_method_array[@]}"; do
                # Echo info
                echo "seed=${seed} c=${c} r=${r} k=${k} opt_method=${opt_method} bo_surrogate=${bo_surrogate}"

                root_dir="logs/opt/celeba/${t}/"
                start_model="logs/train/celeba/smiling/vae/lightning_logs/version_final/checkpoints/last.ckpt"
                pretrained_predictor_file="logs/train/celeba-dialog-predictor/predictor_128.pth.tar"
                scaled_predictor_state_dict="logs/train/celeba-dialog-predictor/predictor_128_scaled3.pth.tar"
                celeba_data_path="data/celeba-dialog"
                # Run the command
                python weighted_retraining/opt_scripts/opt_celeba.py \
                    --seed="$seed" $gpu \
                    --tensor_dir="/content/data_tensors_64" \
                    --property_id=3 \
                    --max_property_value=2 \
                    --train_attr_path="${celeba_data_path}/train_attr_list.txt" \
                    --val_attr_path="${celeba_data_path}/val_attr_list.txt" \
                    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
                    --filename_set_path="${celeba_data_path}/filename_set.pickle" \
                    --attr_file="weighted_retraining/configs/attributes.json" \
                    --query_budget="$query_budget" \
                    --retraining_frequency="$r" \
                    --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/${opt_constraint_strategy}_${n_gmm_components}/c_${c}/seed${seed}_${bo_surrogate}_${opt_method}" \
                    --pretrained_model_file="$start_model" \
                    --pretrained_predictor_file="$pretrained_predictor_file" \
                    --scaled_predictor_state_dict="$scaled_predictor_state_dict" \
                    --weight_type="$weight_type" \
                    --rank_weight_k="$k" \
                    --n_retrain_epochs="$n_retrain_epochs" \
                    --n_init_retrain_epochs="$n_init_retrain_epochs" \
                    --lso_strategy="$lso_strategy" \
                    --n_samples="$n_samples" \
                    --pretrained_model_type="$t" \
                    --n_out="$r" \
                    --n_starts=10 \
                    --opt_method="$opt_method" \
                    --bo_surrogate="$bo_surrogate" \
                    --opt_constraint_threshold="$c" \
                    --opt_constraint_strategy="$opt_constraint_strategy" \
                    --n_gmm_components="$n_gmm_components" \
                    --mode="all" \
                    --batch_size=64
            done
        done
    done
done