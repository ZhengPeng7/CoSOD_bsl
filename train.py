import os
import argparse
import numpy as np
from itertools import cycle
import torch
import torch.optim as optim

from config import Config
from dataset import get_loader
from loss import saliency_structure_consistency, SalLoss

from models.baseline import BSL
from evaluation.valid import validate
from util import Logger, AverageMeter, set_seed


# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--start_epoch',
                    default=1,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--ckpt_dir', default=None, help='Temporary folder')
parser.add_argument('--val_save',
                    default='tmp4val_INS',
                    type=str,
                    help=".")

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
train_loaders = []
training_sets = 'INS-CoS+DUTS_class+coco-seg'
for training_set in training_sets.split('+')[1:2]:
    train_loaders.append(
        get_loader(
            os.path.join(config.root_dir, 'images/{}'.format(training_set)),
            os.path.join(config.root_dir, 'gts/{}'.format(training_set)),
            config.size,
            1,
            max_num=config.batch_size,
            istrain=True,
            shuffle=True,
            num_workers=config.num_workers,
            pin=True
        )
    )

# Multiple dataloader are allowed here,
# shorter ones can choose to pad itself to keep pace with the longest ones.
train_loaders = sorted(train_loaders, key=lambda x: len(x), reverse=True)
for idx_train_loader, train_loader in enumerate(train_loaders):
    # Count how many the longest datasets, suppose lengths = 10, 10, 8, 9
    num_longest_datasets = idx_train_loader + 1
    if idx_train_loader > len(train_loaders) - 2 or len(train_loaders[idx_train_loader + 1]) < len(train_loader):
        break
train_loaders_aligned = [
        train_loaders[idx_train_loader] if idx_train_loader < num_longest_datasets
        else cycle(train_loaders[idx_train_loader]) if config.shorter_data_loader_pad else train_loaders[idx_train_loader]
        for idx_train_loader in range(len(train_loaders))
    ]

test_loaders = {}
for testset in args.testsets.split('+'):
    test_loader = get_loader(
        os.path.join(config.root_dir, 'images', testset), os.path.join(config.root_dir, 'gts', testset),
        config.size, 1, istrain=False, shuffle=False, num_workers=int(config.num_workers//2), pin=True
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
model = BSL().to(config.device)

# Setting optimizer
if config.optimizer == 'Adam':
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[lde if lde > 0 else args.epochs + lde for lde in config.lr_decay_epochs],
    gamma=0.1
)


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
    val_epochs = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss = train_epoch(epoch)
        if config.validation and epoch >= args.epochs - config.val_last and (args.epochs - epoch) % config.save_step == 0:
            measures = validate(model, test_loaders, args.val_save, args.val_sets, valid_only_S=config.valid_only_S)
            for idx_val_set, val_set in enumerate(args.val_sets.split('+')):
                ## To be done for multiple val sets
                # measures[val_set][metric]
                if val_set == 'CoCA':
                    val_measures.append(measures[idx_val_set][0])
                    val_epochs.append(epoch)
                if config.valid_only_S:
                    print(
                        'Validation: S_measure on CoCA for epoch-{} is {:.4f}. '
                        'Best epoch is epoch-{} with S_measure {:.4f}'.format(
                            epoch, measures[idx_val_set][0],
                            val_epochs[np.argmax(val_measures)], np.max(val_measures)
                        )
                    )
                else:
                    metric_scores = {}
                    for k, v in zip(['sm', 'mae', 'fm', 'wfm', 'em'], measures[idx_val_set]):
                        metric_scores[k] = v['curve'].max() if isinstance(v, dict) else v
                    for (metric, score) in metric_scores:
                        if metric == 'sm':
                            print(
                                'Validation: {} on {} for epoch-{} is {:.4f}. '
                                'Best epoch is epoch-{} with S_measure {:.4f}'.format(
                                    metric, val_set, epoch, score,
                                    val_epochs[np.argmax(val_measures)], np.max(val_measures)
                                )
                            )
                        else:
                            print(
                                'Validation: {} on {} for epoch-{} is {:.4f}.'.format(
                                    metric, val_set, epoch, score
                                )
                            )
        # Save checkpoint
        if epoch >= args.epochs - config.val_last and (args.epochs - epoch) % config.save_step == 0:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'ep{}.pth'.format(epoch)))
        lr_scheduler.step()


def train_batch(model, batch, loss_log):
    inputs = batch[0].to(config.device).squeeze(0)
    gts = batch[1].to(config.device).squeeze(0)

    return_values = model(inputs)
    scaled_preds = return_values[0]

    loss = sal_loss(scaled_preds, gts)

    loss_log.update(loss, inputs.size(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train_epoch(epoch):
    loss_log = AverageMeter()
    global logger_loss_idx
    model.train()

    for batch_idx, batchs in enumerate(zip(*train_loaders_aligned)):
        # Do not xxx = zip(*train_loaders)!
        # If you assign it separately before, xxx will stop iteration after its first epoch.
        loss_value = 0.
        for idx_training_set, batch in enumerate(batchs):
            loss_curr = train_batch(model, batch, loss_log)
        loss_value += loss_curr

        with open(logger_loss_file, 'a') as f:
            f.write('step {}, {}\n'.format(logger_loss_idx, loss_value))
        logger_loss_idx += 1
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

        # Logger
        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}]'.format(epoch, args.epochs, batch_idx, len(train_loaders[0]))
            info_loss = 'Train Loss: loss_sal: {:.3f}'.format(loss_value)
            info_loss += ', Loss_total: {loss.val:.3f} ({loss.avg:.3f})  '.format(loss=loss_log)
            logger.info(''.join((info_progress, info_loss)))
    info_loss = '@==Final== Epoch[{0}/{1}]  Train Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=loss_log)
    logger.info(info_loss)

    return loss_log.avg


if __name__ == '__main__':
    main()
