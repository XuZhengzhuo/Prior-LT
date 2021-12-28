import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models

from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from utils import *
from core import *

import argparse
import importlib

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('cfg', type=str, help='training config path')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

best_acc = 0
best_ece = 100


def main():

    args = parser.parse_args()

    # loading config file
    if os.path.exists(args.cfg):
        pth, mdu = args.cfg.rsplit('/')
        mdu = mdu.strip('.py')
        mdu_pth = importlib.import_module(pth)
        cfg = getattr(mdu_pth, mdu)
    else:
        print('No config file available!')
        exit()

    # prepare store fold
    if cfg.debug:
        logger, model_dir = None, None
    else:
        logger, model_dir = create_logger(cfg, args.cfg)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('Seed will turn on the CUDNN deterministic setting')

    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')

    main_worker(logger, model_dir, cfg)


def main_worker(logger, model_dir, cfg):
    global best_acc, best_ece

    if cfg.gpu is not None and torch.cuda.is_available():
        print("Use GPU: {} for training".format(cfg.gpu))

    # create model
    print("=> creating model '{}'".format(cfg.arch))
    num_classes = 100 if cfg.dataset == 'cifar100' else 10
    use_norm = True if cfg.loss_type == 'LDAM' else False
    model = models.__dict__[cfg.arch](num_classes=num_classes,
                                      use_norm=use_norm)

    if cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    # Resume from some ckpt
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume, map_location=f'cuda:{cfg.gpu}')
            cfg.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_ece = checkpoint['best_ece']
            if cfg.gpu is not None:
                best_acc = best_acc.to(cfg.gpu)
            model.load_state_dict(checkpoint['state_dict_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True

    # Data loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if cfg.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data',
                                         imb_type=cfg.imb_type,
                                         imb_factor=cfg.imb_factor,
                                         rand_number=cfg.rand_number,
                                         train=True,
                                         download=True,
                                         transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_val)
    elif cfg.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data',
                                          imb_type=cfg.imb_type,
                                          imb_factor=cfg.imb_factor,
                                          rand_number=cfg.rand_number,
                                          train=True,
                                          download=True,
                                          transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return

    # train label prior
    cfg.train_cls_num_list = np.array(train_dataset.get_cls_num_list())
    # test label distribution
    # assume to be balance
    # test agnostic todo
    cfg.inf_label_distribution = np.ones(shape=cfg.train_cls_num_list.shape)

    print('Train cls num list:')
    print(cfg.train_cls_num_list)
    print('Inference label distribution:')
    print(cfg.inf_label_distribution)

    norm_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=(norm_sampler is None),
                                               num_workers=cfg.workers,
                                               pin_memory=True,
                                               sampler=norm_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=True)

    per_cls_weights = None

    if cfg.loss_type == 'CE':
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(cfg.gpu)

    elif cfg.loss_type == 'Bayias':
        cls_nl = cfg.train_cls_num_list
        inf_ld = cfg.inf_label_distribution
        weight = None
        criterion = Bayias_compensated_loss(train_cls_num_list=cls_nl,
                                            inf_lable_distrbution=inf_ld,
                                            weight=weight).cuda(cfg.gpu)

    else:
        warnings.warn('Loss type is not listed')
        return

    if not cfg.debug:
        logger.info('Training starts ...')

    for epoch in range(cfg.start_epoch, cfg.epochs):

        adjust_learning_rate(optimizer, cfg.lr, epoch)

        # train
        train(train_loader, model, criterion, optimizer, epoch, cfg)

        # evaluate
        acc1, ece = validate(val_loader, model, criterion, epoch, logger, cfg)

        is_best = acc1 > best_acc
        best_acc = max(acc1, best_acc)
        if is_best:
            its_ece = ece

        if not cfg.debug:
            logger.info('Best Prec@1: %.3f%% ECE: %.3f%%\n' %
                        (best_acc, its_ece))

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict_model': model.state_dict(),
                'stage_dict_optim': optimizer.state_dict(),
                'best_acc': best_acc,
                'its_ece': its_ece,
            }, is_best, epoch == cfg.save_ckpt_epoch, model_dir)

    os.rename(model_dir[:-4] + 'logs/log.txt',
              model_dir[:-4] + 'logs/log_%.3f%%.txt' % best_acc)


if __name__ == '__main__':
    main()
