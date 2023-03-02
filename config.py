import os
import math


class Config():
    def __init__(self) -> None:
        self.validation = True
        self.valid_only_S = True
        # Backbone
        self.bb = ['cnn-vgg16', 'cnn-vgg16bn', 'cnn-resnet50', 'trans-pvt'][3]
        self.pvt_weights = ['../bb_weights/pvt_v2_b2.pth', ''][0]
        self.freeze_bb = True

        # Components
        self.consensus = ['', 'GCAM'][0]
        self.dec_blk = ['ResBlk'][0]
        self.dec_att = ['', 'ASPP'][0]
        self.dilation = 1
        self.dec_channel_inter = ['fixed', 'adap'][0]

        # Data loader
        self.shorter_data_loader_pad = True
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:1]  # Augmentation
        self.size = 256
        self.auto_pad = ['', 'adaptive', 'fixed'][0]
            # padding in online batchs.
            # All the padded image will be given a unique preproc, so the preproc is better done if padding chosen.
        self.num_workers = 8
        self.batch_size = 16    # batch size per group
        self.loadN = 2          # load N groups per batch
        # Training
        self.optimizer = ['Adam'][0]
        self.lr = 1e-4 * math.sqrt(self.batch_size / 16)  # adapt the lr linearly
        self.lr_decay_epochs = [1e7]
            # Set to N / -N to decay the lr in the N-th / last N-th epoch, can decay multiple times: [N1, N2, ...].

        # Loss
        losses = ['sal']
        self.loss = losses[:]
        # Loss + Triplet Loss
        self.lambdas_sal_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 30 * 1,          # high performance
            'iou': 0.5 * 1,         # 0 / 255
            'ssim': 1 * 0,          # help contours
            'mse': 150 * 0,         # can smooth the saliency map
            'reg': 100 * 0,
            'triplet': 3 * 0,
        }

        # Others
        self.device = ['cuda', 'cpu'][0]
        self.rand_seed = 7

        # Read the dataset dir, validation range, and validation step from shell script
        with open('go.sh', 'r') as f:
            lines = f.readlines()
            self.root_dir = [l.strip() for l in lines if 'root_dir=' in l][0].split('=')[-1]
            self.val_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
            self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])
