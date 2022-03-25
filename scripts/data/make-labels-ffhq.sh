save_dir="data/ffhq"
data_dir="data/ffhq/data_tensors_256"
pretrained_predictor_file="logs/train/celeba-dialog-predictor/predictor_128.pth.tar"
scaled_predictor_state_dict="logs/train/celeba-dialog-predictor/predictor_128_scaled3.pth.tar"

python src/projected_gan/pg_generate_labels.py \
    --save_dir="$save_dir" \
    --data_dir="$celeba_dir" \
    --attr_file="src/configs/attributes.json" \
    --pretrained_predictor_file="$pretrained_predictor_file" \
    --scaled_predictor_state_dict="$scaled_predictor_state_dict" \
