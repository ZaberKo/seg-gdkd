from tqdm import tqdm
from utils.flops import cal_multi_adds, cal_param_size
from dataset.datasets import CSTrainValSet
from utils.score import SegmentationMetric
from utils.logger import setup_logger
from utils.distributed import *
from utils.config import DictAction
from models.model_zoo import get_segmentation_model
from losses.distiller import get_criterion_distiller
from losses.loss import SegCrossEntropyLoss, CriterionKD, CriterionMiniBatchCrossImagePair
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn as nn
import torch
import argparse
import time
import datetime
import os
import shutil
import sys
import math

import numpy as np
from pathlib import Path
import wandb

from collections import defaultdict

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--student-model', type=str, default='deeplabv3',
                        help='model name')
    parser.add_argument('--student-backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='resnet101',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./data/cityscapes/',
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='ignore label')

    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                        help='w-decay (default: 5e-4)')

    parser.add_argument("--kd-method", type=str, default='dist')
    parser.add_argument("--lambda-kd", type=float,
                        default=1., help="lambda_kd")

    parser.add_argument('--kd-options',
                        nargs='+',
                        action=DictAction,
                        help="key-value pairs of distiller options. eg: KEY=VALUE",
                        default={}
                        )

    # cuda setting
    parser.add_argument('--local_rank', type=int)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--work-dir', type=str, default='./work_dirs/debug')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=400,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=400,
                        help='per iters to val')
    parser.add_argument('--teacher-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str, default='None',
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')

    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')

    # wandb_setting
    parser.add_argument("--wandb-suffix", type=str)

    args = parser.parse_args()

    if args.student_backbone.startswith('resnet'):
        args.aux = True
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    args.device = "cuda"

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda', args.local_rank)
        self.distributed = args.distributed
        self.world_size = args.world_size
        self.local_rank = args.local_rank

        self.logger = setup_logger("semantic_segmentation", args.work_dir, get_rank(),
                                   filename=f'{args.experiment_name}_log.txt')
        self.logger.info("Using {} GPUs".format(args.world_size))
        self.logger.info(vars(args))

        train_dataset = CSTrainValSet(args.data,
                                      list_path='./dataset/list/cityscapes/train.lst',
                                      max_iters=args.max_iterations*args.batch_size,
                                      crop_size=args.crop_size, scale=True, mirror=True)
        val_dataset = CSTrainValSet(args.data,
                                    list_path='./dataset/list/cityscapes/val.lst',
                                    crop_size=(1024, 2048), scale=False, mirror=False)

        train_batch_size = args.batch_size // self.world_size
        train_sampler = make_data_sampler(
            train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(
            train_sampler, train_batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(
            val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=2,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        self.t_model = get_segmentation_model(model=args.teacher_model,
                                              backbone=args.teacher_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base='None',
                                              pretrained=args.teacher_pretrained,
                                              aux=True,
                                              norm_layer=nn.BatchNorm2d,
                                              num_class=train_dataset.num_class)

        self.s_model = get_segmentation_model(model=args.student_model,
                                              backbone=args.student_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base=args.student_pretrained_base,
                                              pretrained='None',
                                              aux=args.aux,
                                              norm_layer=BatchNorm2d,
                                              num_class=train_dataset.num_class)

        self.t_model = self.t_model.to(self.device)
        self.s_model = self.s_model.to(self.device)

        for t_p in self.t_model.parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()

        # resume checkpoint if needed
        # TODO: check whether functional
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.s_model.load_state_dict(torch.load(
                    args.resume, map_location=self.device))

        # create criterion

        self.criterion = SegCrossEntropyLoss(
            ignore_index=args.ignore_label).to(self.device)

        self.criterion_kd = get_criterion_distiller(
            args.kd_method, **args.kd_options)
        self.criterion_kd.to(self.device)

        # params_list = nn.ModuleList([])
        # params_list.append(self.s_model)

        self.optimizer = torch.optim.SGD(self.s_model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(
                self.s_model, device_ids=[args.local_rank])

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.iters = 0
        self.best_metrics = dict(
            pixAcc=0.0,
            mIoU=0.0
        )

    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = 1e-4 + (base_lr - 1e-4)*((1-float(iter)/max_iter)**(power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def reduce_tensor(self, tensor):
        if not dist.is_initialized():
            return tensor

        if not issubclass(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=self.device)
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        rt = self.reduce_tensor(tensor)
        rt /= self.world_size
        return rt

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        self.logger.info('Start training, Total Iterations {:d}'.format(
            args.max_iterations))

        self.s_model.train()

        train_info = defaultdict(list)

        # for iteration, (images, targets, _) in tqdm(
        #         enumerate(self.train_loader, 1), total=len(self.train_loader)):
        for iteration, (images, targets, _) in enumerate(self.train_loader, 1):
            self.iters = iteration

            images = images.to(self.device)
            targets = targets.long().to(self.device)

            with torch.no_grad():
                t_outputs = self.t_model(images)

            s_outputs = self.s_model(images)

            kd_loss = self.args.lambda_kd * \
                self.criterion_kd(s_outputs[0], t_outputs[0], targets)

            if self.args.aux:
                task_loss = self.criterion(
                    s_outputs[0], targets) + 0.4 * self.criterion(s_outputs[1], targets)
                kd_loss += 0.4 * self.args.lambda_kd * \
                    self.criterion_kd(s_outputs[1], t_outputs[1], targets)
            else:
                task_loss = self.criterion(s_outputs[0], targets)

            losses = task_loss + kd_loss
            lr = self.adjust_lr(base_lr=args.lr, iter=iteration-1,
                                max_iter=args.max_iterations, power=0.9)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            task_losses_reduced = self.reduce_mean_tensor(task_loss.detach())
            kd_losses_reduced = self.reduce_mean_tensor(kd_loss.detach())

            train_info['loss_task'].append(task_losses_reduced.item())
            train_info['loss_kd'].append(kd_losses_reduced.item())

            if is_main_process():
                if iteration % log_per_iters == 0:
                    cost_time_str = str(datetime.timedelta(
                        seconds=int(time.time() - start_time)))
                    eta_str = str(datetime.timedelta(seconds=int((
                        (time.time() - start_time) / iteration) *
                        (args.max_iterations - iteration)
                    )))
                    lr = self.get_lr()
                    task_loss = np.array(train_info['loss_task']).mean()
                    kd_loss = np.array(train_info['loss_kd']).mean()
                    # reset train_info
                    train_info = defaultdict(list)

                    self.logger.info(
                        f"Iters: {iteration:d}/{args.max_iterations:d} || Lr: {lr:.6f} "
                        f"|| Task Loss: {task_loss:.4f} || KD Loss: {kd_loss:.4f} "
                        f"|| Cost Time: {cost_time_str} || Estimated Time: {eta_str}"
                    )

                    wandb.log(dict(
                        lr=lr,
                        loss_task=task_loss,
                        loss_kd=kd_loss
                    ), step=iteration)

                if iteration % save_per_iters == 0 and save_to_disk:
                    save_checkpoint(self.s_model, self.args,
                                    cur_iter=iteration, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.s_model.train()

        if is_main_process():
            save_checkpoint(self.s_model, self.args, is_best=False)

            total_training_time = time.time() - start_time
            total_training_str = str(
                datetime.timedelta(seconds=total_training_time))
            self.logger.info(
                f"Total training time: {total_training_str} ({total_training_time / args.max_iterations:.4f}s / it)"
            )
            self.logger.info(
                f"Best Val PixACC {self.best_metrics['pixAcc']:.3f}, mIoU: {self.best_metrics['mIoU']:.3f}"
            )

    def validation(self):
        self.metric.reset()

        torch.cuda.empty_cache()  # TODO: check if it helps
        self.s_model.eval()
        self.logger.info("Start validation, Total sample: {:d}".format(
            len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader, 1):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = self.s_model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(
                outputs[0], (H, W), mode='bilinear', align_corners=True)

            target = target.view(H * W)
            eval_index = target != args.ignore_label

            full_probs = outputs[0].view(
                1, self.metric.nclass, H * W)[:, :, eval_index].unsqueeze(-1)
            target = target[eval_index].unsqueeze(0).unsqueeze(-1)

            self.metric.update(full_probs, target)
            pixAcc, mIoU = self.metric.get()
            self.logger.info(
                f"Sample: {i:d}, Validation pixAcc: {pixAcc*100:.3f}, mIoU: {mIoU*100:.3f}")

        if self.distributed:
            sum_total_correct = torch.tensor(
                self.metric.total_correct).to(self.device)
            sum_total_label = torch.tensor(
                self.metric.total_label).to(self.device)
            sum_total_inter = torch.tensor(
                self.metric.total_inter).to(self.device)
            sum_total_union = torch.tensor(
                self.metric.total_union).to(self.device)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            eps = 2.220446049250313e-16
            pixAcc = (1.0 * sum_total_correct / (eps + sum_total_label)).item()
            IoU = 1.0 * sum_total_inter / (eps + sum_total_union)
            mIoU = IoU.mean().item()

        if is_main_process():
            self.logger.info(
                f"Overall validation pixAcc: {pixAcc*100:.3f}, mIoU: {mIoU*100:.3f}")
            wandb.log(dict(
                pixAcc=pixAcc,
                mIoU=mIoU
            ), step=self.iters)

        self.best_metrics['pixAcc'] = max(mIoU, self.best_metrics['pixAcc'])

        if mIoU > self.best_metrics['mIoU']:
            self.best_metrics['mIoU'] = mIoU
            if is_main_process():
                save_checkpoint(self.s_model, self.args,
                                self.iters, is_best=True)
        synchronize()


def save_checkpoint(model, args, cur_iter, is_best=False):
    """Save Checkpoint"""
    directory = Path(args.work_dir).expanduser()

    filename = directory/f"{args.experiment_name}_iter{cur_iter}.pth"

    if args.distributed:
        model = model.module

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = directory/f"{args.experiment_name}_best_model.pth"
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    args.world_size = int(os.environ.get("WORLD_SIZE", '1'))

    if args.local_rank is None:
        # support torchrun
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    args.distributed = args.world_size > 1

    if args.distributed:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
        synchronize()

    args.experiment_name = f'{args.kd_method}_{args.student_model}-{args.student_backbone}_{args.teacher_model}-{args.teacher_backbone}'

    if is_main_process():
        work_dir = Path(args.work_dir).expanduser()
        if not work_dir.exists():
            os.makedirs(args.work_dir)

        wandb_name = args.experiment_name
        wandb_tags = [args.kd_method, f'{args.student_model}-{args.student_backbone}',
                      f'{args.teacher_model}-{args.teacher_backbone}', 'cityscapes']

        suffix_list = []
        if args.kd_options is not None:
            suffix_list = [f"{k}={v}" for k, v in args.kd_options.items()]

        if args.wandb_suffix:
            suffix_list.append(args.wandb_suffix)
            wandb_tags.append(args.wandb_suffix)

        if len(suffix_list):
            suffix_str = ",".join(suffix_list)
            wandb_name = f'{wandb_name}|{suffix_str}'

        # setup wandb
        wandb.init(
            project='semseg_distill',
            name=wandb_name,
            group=wandb_name+'_group',
            config=vars(args),
            tags=wandb_tags,
            dir=args.work_dir
        )
        wandb.define_metric('pixAcc', summary='max')
        wandb.define_metric('mIoU', summary='max')

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
