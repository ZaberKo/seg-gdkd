from __future__ import print_function
from collections import defaultdict
from utils.flops import cal_multi_adds, cal_param_size
from dataset import get_val_dataset
from utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from utils.logger import setup_logger
from utils.visualize import get_color_pallete
from utils.score import SegmentationMetric
from models.model_zoo import get_segmentation_model
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
from tqdm import tqdm

import os
import sys
import argparse
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation validation With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/cityscapes/',
                        help='dataset directory')
    parser.add_argument('--data-list', type=str, default='./dataset/list/cityscapes/val.lst',
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[1024, 2048], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')

    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default='psp_resnet18_citys_best_model.pth',
                        help='pretrained seg model')
    parser.add_argument('--save-dir', default='../runs/logs/',
                        help='Directory for saving predictions')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='input batch size for training (default: 8)')
    # validation
    parser.add_argument(
        '--scales', default=[1.], type=float, nargs='+', help='multiple scales')
    parser.add_argument('--flip-eval', action='store_true', default=False,
                        help='flip_evaluation')
    args = parser.parse_args()

    if args.backbone.startswith('resnet'):
        args.aux = True
    elif args.backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args


class Evaluator(object):
    def __init__(self, args, num_gpus):
        self.args = args
        self.num_gpus = num_gpus
        self.device = torch.device(args.device)

        # dataset and dataloader
        self.val_dataset = get_val_dataset(args)

        val_sampler = make_data_sampler(
            self.val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(
            val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model,
                                            backbone=args.backbone,
                                            aux=args.aux,
                                            pretrained=args.pretrained,
                                            pretrained_base='None',
                                            local_rank=0,
                                            norm_layer=BatchNorm2d,
                                            num_class=self.val_dataset.num_class).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logger.info('Params: %.2fM FLOPs: %.2fG'
                        % (cal_param_size(self.model) / 1e6, cal_multi_adds(self.model, (1, 3, 1024, 2048))/1e9))

        self.model.to(self.device)

        self.metric = SegmentationMetric(self.val_dataset.num_class)

    def predict_whole(self, net, image):

        prediction = net(image.cuda())
        if isinstance(prediction, tuple) or isinstance(prediction, list):
            prediction = prediction[0]

        return prediction

    def eval(self):
        self.metric.reset()
        self.model.eval()

        model = self.model

        num_classes = self.val_dataset.num_class

        pred_logits = defaultdict(list)

        logger.info("Start validation, Total sample: {:d}".format(
            len(self.val_loader)))
        for i, (image, target, filename) in tqdm(
            enumerate(self.val_loader), total=len(self.val_loader)):
            image = image.to(self.device)
            target = target.long().to(self.device)

            N_, C_, H_, W_ = image.size()
            tile_size = (H_, W_)

            with torch.no_grad():
                logits = self.predict_whole(model, image)

            h, w = logits.shape[2:]
            target_down = target = F.interpolate(
                target.float().unsqueeze(1), (h, w), mode='nearest').squeeze(1)
            
            flat_logits = logits.permute(0,2,3,1).reshape(-1, num_classes)
            flat_target = target_down.long().flatten()

            for i in range(num_classes):
                eval_index = (flat_target == i)
                cls_logits = flat_logits[eval_index,:]
                pred_logits[i].append(cls_logits.cpu())

        for i in pred_logits.keys():
            pred_logits[i] = torch.cat(pred_logits[i], dim=0).numpy()
            print(f"{i}:", pred_logits[i].shape)

        results={
            str(k): v for k,v in pred_logits.items()
        }

        np.savez(os.path.join(args.outdir, 'pred_logits.npz'), **results)


if __name__ == '__main__':
    args = parse_args()
    num_gpus = 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"

    # TODO: optim code
    outdir = '{}_{}_{}'.format(args.model, args.backbone, args.dataset)
    args.outdir = os.path.join(args.save_dir, outdir)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    logger = setup_logger("semantic_segmentation", args.save_dir, get_rank(),
                          filename='{}_{}_{}_multiscale_val.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args, num_gpus)
    evaluator.eval()
    torch.cuda.empty_cache()
