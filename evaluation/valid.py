import os
from glob import glob
import shutil
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

from config import Config

import evaluation.metrics as Measure


config = Config()
def validate(model, test_loaders, val_dir, testsets='CoCA', valid_only_S=True):
    model.eval()

    testsets = testsets.split('+')
    measures = []
    print('Validating ...', end='')
    for testset in testsets[:]:
        print(', ' + testset, end='')
        test_loader = test_loaders[testset]
        
        saved_root = os.path.join(val_dir, testset)
        if os.path.exists(saved_root):
            shutil.rmtree(saved_root)

        for idx_batch, batch in enumerate(test_loader):
            # one batch contains all images of one class
            # if idx_batch >= 1:
            #     continue
            inputs = batch[0].to(config.device).squeeze(0)
            gts = batch[1].to(config.device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs)[-1]

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True).sigmoid()
                save_tensor_img(res, os.path.join(saved_root, subpath))

            pred_pth_lst = glob(os.path.join(val_dir, testset, '*', '*'))
            gt_pth_lst = []
            for p in pred_pth_lst:
                p = p.replace(val_dir, os.path.join(config.root_dir, 'gts'))
                if not os.path.exists(p):
                    p = ''.join((p[:-4], '.jpg' if p[-4:] == '.png' else '.png'))
                gt_pth_lst.append(p)
            s_measure = evaluator(gt_pth_lst, pred_pth_lst, only_S=valid_only_S)
            measures.append(s_measure)
    print()
    model.train()
    return measures


def evaluator(gt_pth_lst, pred_pth_lst, only_S=True):
    # define measures
    SM = Measure.Smeasure()
    if not only_S:
        FM = Measure.Fmeasure()
        WFM = Measure.WeightedFmeasure()
        EM = Measure.Emeasure()
        MAE = Measure.MAE()

    assert len(gt_pth_lst) == len(pred_pth_lst)

    for idx in range(len(gt_pth_lst)):
        gt_pth = gt_pth_lst[idx]
        pred_pth = pred_pth_lst[idx]

        pred_pth = pred_pth[:-4] + '.png'
        if os.path.exists(pred_pth):
            pred_ary = cv2.imread(pred_pth, cv2.IMREAD_GRAYSCALE)
        else:
            pred_ary = cv2.imread(pred_pth.replace('.png', '.jpg'), cv2.IMREAD_GRAYSCALE)
        gt_ary = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)
        pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

        SM.step(pred=pred_ary, gt=gt_ary)
        if not only_S:
            FM.step(pred=pred_ary, gt=gt_ary)
            WFM.step(pred=pred_ary, gt=gt_ary)
            EM.step(pred=pred_ary, gt=gt_ary)
            MAE.step(pred=pred_ary, gt=gt_ary)

    sm = SM.get_results()['sm']
    if not only_S:
        fm = FM.get_results()['fm']
        wfm = WFM.get_results()['wfm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']
        return fm, wfm, sm, em, mae
    else:
        return [sm]


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)
