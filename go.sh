#!/bin/bash
# Run script
method="$1"
epochs=1

# Train
CUDA_VISIBLE_DEVICES=0 python train.py --ckpt_dir ckpt/${method} --epochs ${epochs}


# # Test & Eval
# # If dirs of performance and predictions already exist, remove them.
# if [ "${method}" ] ;then
#     rm -rf evaluation/${method}
#     rm -rf /root/autodl-tmp/datasets/sod/preds/${method}
# fi

# val_last=30
# step=10
# for ((ep=${epochs};ep>${epochs}-${val_last};ep-=${step}))
# do
# pred_dir=/root/autodl-tmp/datasets/sod/preds/${method}/ep${ep}
# # [ ${ep} -gt $[${epochs}-${val_last}] ] && CUDA_VISIBLE_DEVICES=$2 python evaluation/main.py --model_dir ${method}/ep$[${ep}-${step}] --txt_name ${method}; \
# CUDA_VISIBLE_DEVICES=$2 python test.py --pred_dir ${pred_dir} --ckpt ckpt/${method}/ep${ep}.pth --size ${size}
# done

# # python evaluation/main.py --model_dir ${method}/ep$[${ep}-${step}] --txt_name ${method}
# python evaluation/main.py --model_dir ${method} --txt_name ${method}

# nvidia-smi
# hostname
