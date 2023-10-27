"""
ImageNet Training Script
This script is adapted from pytorch-image-models by Ross Wightman (https://github.com/rwightman/pytorch-image-models/)
It was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)
"""
import argparse
import datetime
import numpy as np
import time
import logging
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
import timm
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, distribute_bn

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
# import models
import svt
import utils
import collections

from tlt.data import create_token_label_target, TokenLabelMixup, FastCollateTokenLabelMixup, \
    create_token_label_loader, create_token_label_dataset
from loss import TokenLabelGTCrossEntropy, TokenLabelCrossEntropy, TokenLabelSoftTargetCrossEntropy

from util.flops_counter import get_model_complexity_info
from util.checkpoint_saver import CheckpointSaver2

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--config', required=True, type=str, help='config')

    # Model parameters
    parser.add_argument('--model', default='pvt_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # parser.add_argument('--model-ema', action='store_true')
    # parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    # parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    # parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    # parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
    #                     help='Name of teacher model to train (default: "regnety_160"')
    # parser.add_argument('--teacher-path', type=str, default='')
    # parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    # parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    # parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--use-mcloader', action='store_true', default=False, help='Use mcloader')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    ###############################################################################################################
    # Token labeling
    parser.add_argument('--token-label', action='store_true', default=False,
                    help='Use dense token-level label map for training')
    parser.add_argument('--token-label-data', type=str, default='', metavar='DIR',
                    help='path to token_label data')
    parser.add_argument('--token-label-size', type=int, default=1, metavar='N',
                    help='size of result token label map')
    parser.add_argument('--dense-weight', type=float, default=0.5,
                    help='Token labeling loss multiplier (default: 0.5)')
    parser.add_argument('--cls-weight', type=float, default=1.0,
                    help='Cls token prediction loss multiplier (default: 1.0)')

    parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
    ###############################################################################################################
    return parser

def build_no_token_label(args):
    print("Creating dataloader for build_no_token_label")
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    return dataset_train, data_loader_train, dataset_val, data_loader_val, mixup_fn, args.nb_classes


def build_token_label(args):
    print("Creating dataloader for build_token_label")
    dataset_train = create_token_label_dataset(
            '', root=args.data_path, label_root=args.token_label_data)
    dataset_eval = timm.data.create_dataset('',
            root=args.data_path,
            split='validation',
            is_training=False,
            batch_size=int(1.5 * args.batch_size))

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    args.mixup = 0.0
    args.cutmix = 0.0
    args.cutmix_minmax = None
    args.train_interpolation = 'random'
    args.prefetcher = True
    num_aug_splits = 0
    args.pin_mem = False

    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.token_label and args.token_label_data:
        use_token_label = True
    else:
        use_token_label = False
    
    loader_train = create_token_label_loader(
        dataset_train,
        input_size=(3, args.input_size, args.input_size),
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=timm.data.constants.IMAGENET_DEFAULT_MEAN,
        std=timm.data.constants.IMAGENET_DEFAULT_STD,
        num_workers=args.num_workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        use_token_label=use_token_label)

    loader_eval = timm.data.create_loader(
        dataset_eval,
        input_size=(3, args.input_size, args.input_size),
        batch_size=int(1.5 * args.batch_size),
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation='bicubic',
        mean=timm.data.constants.IMAGENET_DEFAULT_MEAN,
        std=timm.data.constants.IMAGENET_DEFAULT_STD,
        num_workers=args.num_workers,
        distributed=args.distributed,
        crop_pct=0.96,
        pin_memory=args.pin_mem,
        persistent_workers=False
    )
    return dataset_train, loader_train, dataset_eval, loader_eval, mixup_fn, 1000

def build_imagenet_dataset(args):
    if args.token_label_data:
        return build_token_label(args)
    else:
        return build_no_token_label(args)

def main(args):
    timm.utils.setup_default_logging()
    utils.init_distributed_mode(args)
    
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    utils.setup_logger(args.output_dir, distributed_rank=utils.get_rank())
    _logger = logging.getLogger('train')
    _logger.info(args)

    dataset_train, data_loader_train, dataset_val, data_loader_val, mixup_fn, args.nb_classes \
        = build_imagenet_dataset(args)

    _logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        token_label=args.token_label,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']['state_dict']
        else:
            checkpoint_model = checkpoint['state_dict']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                _logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)
    print('Model Architect',model)

    #with torch.cuda.amp.autocast(enabled=True):
    #    flops_count, params_count = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    #        print_per_layer_stat=False, verbose=False)
    #print(f'Model %s created, flops_count: %s, param count: %s' % (args.model, flops_count, params_count))

    model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model_ema_decay,
    #         device='cpu' if args.model_ema_force_cpu else '',
    #         resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info('number of params: ' + str(n_parameters))

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.token_label:
        criterion = TokenLabelCrossEntropy(dense_weight=args.dense_weight, \
            cls_weight=args.cls_weight, mixup_active=False).cuda()
    else:
        criterion = LabelSmoothingCrossEntropy()
        if args.mixup > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        criterion = DistillationLoss(
            criterion, None, 'none', 0, 0
        )

    ##########################################################################################
    saver = None
    best_metric = None
    best_epoch = None
    if utils.get_rank() == 0:
        decreasing = False
        saver = CheckpointSaver2(model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=args.output_dir,
            recovery_dir=args.output_dir,
            decreasing=decreasing,
            max_history=10)
    ##########################################################################################

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            msg = model_without_ddp.load_state_dict(checkpoint['model']['state_dict'])
        else:
            msg = model_without_ddp.load_state_dict(checkpoint['state_dict'])
        _logger.info(msg)
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            # if args.model_ema:
            #     utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        _logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    _logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            fp32=args.fp32_resume, args=args
        )

        if args.distributed:
            _logger.info('Distributing BatchNorm running means and vars')
            distribute_bn(model, utils.get_world_size(), True)

        lr_scheduler.step(epoch)
        #if args.output_dir:
        #    checkpoint_paths = [output_dir / 'checkpoint.pth']
        #    for checkpoint_path in checkpoint_paths:
        #        utils.save_on_master({
        #            'model': model_without_ddp.state_dict(),
        #            'optimizer': optimizer.state_dict(),
        #            'lr_scheduler': lr_scheduler.state_dict(),
        #            'epoch': epoch,
        #            # 'model_ema': get_state_dict(model_ema),
        #            'scaler': loss_scaler.state_dict(),
        #            'args': args,
        #        }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)

        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = test_stats['acc1']
            best_metric, best_epoch = saver.save_checkpoint(
                epoch, metric=save_metric)
        if best_metric is not None:
            _logger.info('*** Best metric: {0} (epoch {1})'.format(
                best_metric, best_epoch))

        _logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        _logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "tlog.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        _logger.info(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    _logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
