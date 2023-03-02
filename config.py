import os


class Config():
    def __init__(self) -> None:
        self.root_dir = '/root/autodl-tmp/datasets/sod'
        self.valid_only_S = True
        # Backbone
        self.bb = ['cnn-vgg16', 'cnn-vgg16bn', 'cnn-resnet50', 'trans-pvt'][3]
        self.pvt_weights = ['../bb_weights/pvt_v2_b2.pth', ''][0]
        # BN
        self.use_bn = self.bb not in ['cnn-vgg16']
        # Augmentation
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]

        # Components
        self.consensus = ['', 'GCAM', 'GWM', 'SGS'][1]
        self.dec_blk = ['ResBlk'][0]
        self.GCAM_metric = ['online', 'offline', ''][0] if self.consensus else ''
        # Training
        self.batch_size = 16
        self.loadN = 2
        self.dec_att = ['', 'ASPP'][0]
        self.auto_pad = ['', 'adaptive', 'fixed'][2]
        self.optimizer = ['Adam'][0]
        self.lr = 1e-4
        self.freeze = True
        self.lr_decay_epochs = [10000]    # Set to negative N to decay the lr in the last N-th epoch.
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

        self.validation = True
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'go' in f and '.sh' in f] + [os.path.join('..', f) for f in os.listdir('..') if 'gco' in f and '.sh' in f]
        with open(run_sh_file[0], 'r') as f:
            lines = f.readlines()
            self.val_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
            self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])