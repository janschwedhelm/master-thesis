save_dir="data/ffhq/data_tensors_256"
celeba_dir="/Users/jani/Downloads/images256x256"
mkdir -p "$save_dir"
python src/celeba/celeba_classifier_dataset.py \
    --save_dir="$save_dir" \
    --celeba_dir="$celeba_dir"