method="$1"
epochs=120
val_last=70
step=10
root_dir=/root/datasets/sod

# Train & val
CUDA_VISIBLE_DEVICES=$2 python train.py --ckpt_dir ckpt/${method} --epochs ${epochs}

nvidia-smi
hostname
