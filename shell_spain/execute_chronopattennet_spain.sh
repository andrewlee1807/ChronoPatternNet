#
# Copyright (c) 2022-2023 Andrew
# Email: andrewlee1807@gmail.com
#

for i in 1 2 3 4 5 6 7 8 9 10 15 20 24 36 48 60 72 84 96 108 120 132; do
   echo "Starting to train model with output length = $i"
   python main.py \
    --dataset_name="spain" \
    --write_log_file=True \
    --model_name="Chrono" \
    --config_path="config/chrono/spain.yaml" \
    --output_length=$i \
    --device=0 \
    --output_dir="benchmark/exp/spain/chrono"
   echo "Finished training model with output length = $i"
   echo "=================================================="
done