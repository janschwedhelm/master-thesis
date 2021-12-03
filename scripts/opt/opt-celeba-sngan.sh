# Meta flags
seed_array=( 1 )
bo_surrogate_array=( DNGO )
opt_method_array=( sampling )
attribute_array=( 0 )
query_budget=500
n_retrain_steps=1
n_init_retrain_steps=1
sample_size=100000
k=1e-3
r=50
weight_type="rank"
lso_strategy="opt"

for seed in "${seed_array[@]}"; do
    for bo_surrogate in "${bo_surrogate_array[@]}"; do
        for opt_method in "${opt_method_array[@]}"; do
            for attribute in "${attribute_array[@]}"; do
                # Echo info
                echo "seed=${seed} r=${r} k=${k} opt_method=${opt_method} bo_surrogate=${bo_surrogate}"
    
                root_dir="logs/opt/celeba/${attribute}/sn-gan/z_128"
                start_model_netg="logs/train/celeba/${attribute}/sn-gan/z_128/max_prop_2/checkpoints/netG/netG_80000_steps.pth"
                start_model_netd="logs/train/celeba/${attribute}/sn-gan/z_128/max_prop_2/checkpoints/netD/netD_80000_steps.pth"
                pretrained_predictor_file="logs/train/celeba-dialog-predictor/predictor_128.pth.tar"
                scaled_predictor_state_dict="logs/train/celeba-dialog-predictor/predictor_128_scaled1.pth.tar"
                celeba_data_path="data/celeba-dialog"
                sample_distribution="normal"
                pretrained_model_prior="normal"
                # Run the command
                python weighted_retraining/opt_scripts/opt_celeba_sngan.py \
                    --seed="$seed" \
                    --tensor_dir="/content/data_tensors_64" \
                    --property_id="$attribute" \
                    --max_property_value=2 \
                    --train_attr_path="$celeba_data_path/train_attr_list.txt" \
                    --val_attr_path="$celeba_data_path/val_attr_list.txt" \
                    --combined_annotation_path="$celeba_data_path/combined_annotation.txt" \
                    --filename_set_path="$celeba_data_path/filename_set.pickle" \
                    --attr_file="weighted_retraining/configs/attributes.json" \
                    --query_budget="$query_budget" \
                    --retraining_frequency="$r" \
                    --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/max_prop_2/seed${seed}_test2" \
                    --pretrained_netg_model_file="$start_model_netg" \
                    --pretrained_netd_model_file="$start_model_netd" \
                    --pretrained_model_prior="$pretrained_model_prior" \
                    --pretrained_predictor_file="$pretrained_predictor_file" \
                    --scaled_predictor_state_dict="$scaled_predictor_state_dict" \
                    --weight_type="$weight_type" \
                    --rank_weight_k="$k" \
                    --n_retrain_steps="$n_retrain_steps" \
                    --n_init_retrain_steps="$n_init_retrain_steps" \
                    --sample_distribution="$sample_distribution" \
                    --opt_method="$opt_method" \
                    --bo_surrogate="$bo_surrogate" \
                    --mode="all" \
                    --batch_size=64 \
                    --n_samples="$sample_size" \
                    --n_best_points="$sample_size" \
                    --n_rand_points=0 \
                    --opt_constraint_strategy="discriminator" \
                    --opt_constraint_threshold=0.5
            done
        done
    done
done
