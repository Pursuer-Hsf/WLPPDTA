nohup python3 train.py \
    --data_path data \
    --results_dir results \
    --fold 1 \
    --mode 0 \
    > log/train_$(date +%y%m%d_%H%M%S).log 2>&1 &