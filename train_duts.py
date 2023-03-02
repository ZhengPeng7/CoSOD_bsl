from random import seed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from util import Logger, AverageMeter, set_seed
import os
import argparse
from dataset import get_loader

import torch.nn.functional as F

from config import Config
from loss import saliency_structure_consistency, SalLoss
from util import generate_smoothed_gt

from models.GCoNet import GCoNet

from evaluation.valid import validate


# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model',
                    default='GCoNet',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--start_epoch',
                    default=1,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--size',
                    default=256,
                    type=int,
                    help='input size')
parser.add_argument('--ckpt_dir', default=None, help='Temporary folder')

parser.add_argument('--val_sets',
                    default='CoCA',
                    type=str,
                    help="Options: 'CoCA+CoSal2015+CoSOD3k'")

parser.add_argument('--testsets',
                    default='CoCA+CoSOD3k+CoSal2015',
                    type=str,
                    help="Options: 'CoCA', 'CoSal2015', 'CoSOD3k'")

args = parser.parse_args()


config = Config()

# Prepare dataset
trainset = 'DUTS'
if 'DUTS' in trainset.split('+'):
    train_img_path = os.path.join(config.root_dir, 'images/DUTS_class')
    train_gt_path = os.path.join(config.root_dir, 'gts/DUTS_class')
    train_loader = get_loader(
        train_img_path,
        train_gt_path,
        args.size,
        1,
        max_num=config.batch_size,
        istrain=True,
        shuffle=True,
        num_workers=0,
        pin=True
    )


test_loaders = {}
for testset in args.testsets.split('+'):
    test_loader = get_loader(
        os.path.join('/root/autodl-tmp/datasets/sod', 'images', testset), os.path.join('/root/autodl-tmp/datasets/sod', 'gts', testset),
        args.size, 1, istrain=False, shuffle=False, num_workers=0, pin=True
    )
    test_loaders[testset] = test_loader

if config.rand_seed:
    set_seed(config.rand_seed)

# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_file = os.path.join(args.ckpt_dir, "log_loss.txt")
logger_loss_idx = 1

# Init model
device = torch.device("cuda")

model = GCoNet().to(device)

# Setting optimizer
if config.optimizer == 'AdamW':
    optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
elif config.optimizer == 'Adam':
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[lde if lde > 0 else args.epochs + lde for lde in config.lr_decay_epochs],
    gamma=0.1
)

# Why freeze the backbone?...
if config.freeze:
    for key, value in model.named_parameters():
        if 'bb.' in key:
            value.requires_grad = False


# log model and optimizer params
# logger.info("Model details:")
# logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
logger.info(lr_scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
sal_loss = SalLoss()


def main():
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(torch.load(args.resume))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    val_measures = []
    for epoch in range(args.start_epoch, args.epochs+1):
        train_loss = train(epoch)
        if config.validation and epoch > args.epochs-1:
            measures = validate(model, test_loaders, args.val_sets)
            val_measures.append(measures)
            print('Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with S_measure {:.4f}'.format(
                epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()), np.max(np.array(val_measures)[:, 0]))
            )
        # Save checkpoint
        if epoch >= args.epochs - config.val_last and (args.epochs - epoch) % config.save_step == 0:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'ep{}.pth'.format(epoch)))
        lr_scheduler.step()


def train(epoch):
    loss_log = AverageMeter()
    global logger_loss_idx
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)

        return_values = model(inputs)
        scaled_preds = return_values[0]

        loss = sal_loss(scaled_preds, gts)

        loss_log.update(loss, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        with open(logger_loss_file, 'a') as f:
            f.write('step {}, {}\n'.format(logger_loss_idx, loss))
        logger_loss_idx += 1
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

        # Logger
        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}]'.format(epoch, args.epochs, batch_idx, len(train_loader))
            info_loss = 'Train Loss: loss_sal: {:.3f}'.format(loss)
            info_loss += ', Loss_total: {loss.val:.3f} ({loss.avg:.3f})  '.format(loss=loss_log)
            logger.info(''.join((info_progress, info_loss)))
    info_loss = '@==Final== Epoch[{0}/{1}]  Train Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=loss_log)
    logger.info(info_loss)

    return loss_log.avg


if __name__ == '__main__':
    main()
