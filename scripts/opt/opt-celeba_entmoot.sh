# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed_array=( 1 )
query_budget=500
n_retrain_epochs=0.1
n_init_retrain_epochs=1

k=1e-3
r=5
weight_type="rank"
lso_strategy="opt"

for seed in "${seed_array[@]}"; do
    # Echo info
    echo "seed=${seed} r=${r} k=${k}"

    root_dir="logs/opt/celeba/smiling/vq-vae"
    start_model="logs/train/celeba/smiling/vq-vae/lightning_logs/version_5/checkpoints/last.ckpt"
    pretrained_predictor_file="logs/train/celeba-dialog-predictor/predictor_128.pth.tar"
    scaled_predictor_state_dict="logs/train/celeba-dialog-predictor/predictor_128_scaled3.pth.tar"
    celeba_data_path="data/celeba-dialog"
    # Run the command
    python weighted_retraining/opt_scripts/opt_celeba_entmoot.py \
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
        --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
        --pretrained_model_file="$start_model" \
        --pretrained_predictor_file="$pretrained_predictor_file" \
        --scaled_predictor_state_dict="$scaled_predictor_state_dict" \
        --weight_type="$weight_type" \
        --rank_weight_k="$k" \
        --n_retrain_epochs="$n_retrain_epochs" \
        --n_init_retrain_epochs="$n_init_retrain_epochs" \
        --lso_strategy="$lso_strategy" \
        --n_out=5 \
        --mode="all" \
        --batch_size=128
done
