method="$1"
epochs=250
iterations=0
val_last=150
step=10
root_dir=/root/datasets/sod

# Train & val
CUDA_VISIBLE_DEVICES=$2 python train.py --ckpt_dir ckpt/${method}

nvidia-smi
hostname
