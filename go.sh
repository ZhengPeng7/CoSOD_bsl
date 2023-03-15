method="$1"
epochs=150
iterations=0
val_last=100
step=10
root_dir=/mnt/workspace/workgroup/mohe/datasets/sod

# Train & val
CUDA_VISIBLE_DEVICES=$2 python train.py --ckpt_dir ckpt/${method}

# Plot scores
python plot_val_scores.py nohup.out

# nvidia-smi
# hostname
