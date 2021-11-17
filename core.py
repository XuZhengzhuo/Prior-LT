import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import choice

from utils import *


class Bayias_compensated_loss(nn.Module):
    def __init__(self,
                 train_cls_num_list=None,
                 inf_lable_distrbution=None,
                 weight=None):
        super(Bayias_compensated_loss, self).__init__()

        self.weight = weight

        self.train_cnl = train_cls_num_list
        self.prior = np.log(self.train_cnl / sum(self.train_cnl))
        self.prior = torch.from_numpy(self.prior).type(torch.cuda.FloatTensor)

        self.inf = inf_lable_distrbution
        self.inf = np.log(self.inf / sum(self.inf))
        self.inf = torch.from_numpy(self.inf).type(torch.cuda.FloatTensor)

    def forward(self, x, target):
        logits = x + self.prior - self.inf
        loss = F.cross_entropy(logits,
                               target,
                               weight=self.weight,
                               reduction='none')
        return loss


def unimix_sampler(batch_size, labels, cls_num_list, tau):
    idx = np.linspace(0, batch_size - 1, batch_size)
    cls_num = np.array(cls_num_list)
    idx_prob = cls_num[labels.cpu().numpy()]
    idx_prob = np.power(idx_prob, tau, dtype=float)
    idx_prob = idx_prob / np.sum(idx_prob)
    idx = choice(idx, batch_size, p=idx_prob)
    idx = torch.Tensor(idx).type(torch.LongTensor)
    return idx


def unimix_factor(labels_1, labels_2, cls_num_list, alpha):
    cls_num_list = np.array(cls_num_list)
    n_i = cls_num_list[labels_1.cpu().numpy()]
    n_j = cls_num_list[labels_2.cpu().numpy()]
    lam = n_j / (n_i + n_j)
    lam = [np.random.beta(alpha, alpha) + t for t in lam]
    lam = np.array([t - 1 if t > 1 else t for t in lam])
    return torch.Tensor(lam).cuda()


def unimix(images, labels, cls_num_list, alpha, tau):

    batch_size = images.size()[0]

    index = unimix_sampler(batch_size, labels, cls_num_list, tau)
    images_1, images_2 = images, images[index, :]
    labels_1, labels_2 = labels, labels[index]

    lam = unimix_factor(labels_1, labels_2, cls_num_list, alpha)

    mixed_images = torch.zeros_like(images)
    for i, s in enumerate(lam):
        mixed_images[i, :, :, :] = images_1[i, :, :, :] * s + images_2[
            i, :, :, :] * (1 - s)
    mixed_images = mixed_images[:batch_size].cuda()

    labels_1, labels_2 = labels_1, labels_2[:batch_size]

    return mixed_images, labels_1, labels_2, lam


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')

    # switch to train mode
    model.train()

    end = time.time()
    for (images, labels) in train_loader:

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(cfg.gpu, non_blocking=True)
            labels = labels.cuda(cfg.gpu, non_blocking=True)

        if cfg.mix_type != None and epoch < cfg.mix_stop_epoch:

            if cfg.mix_type == 'unimix':
                mix_images, lab_1, lab_2, lam = unimix(
                    images=images,
                    labels=labels,
                    cls_num_list=cfg.train_cls_num_list,
                    alpha=cfg.unimix_alp,
                    tau=cfg.unimix_tau)
            else:
                print('Should mixup training but no mix type is selected!')
                os._exit(0)

            output = model(mix_images)
            loss = lam * criterion(output, lab_1) + \
                    (1 - lam) * criterion(output, lab_2)
        else:
            output = model(images)
            loss = criterion(output, labels)

        loss = torch.mean(loss)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


def validate(val_loader, model, criterion, epoch, logger, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    class_num = torch.zeros(cfg.num_classes).cuda()
    correct = torch.zeros(cfg.num_classes).cuda()

    cfd = np.array([])
    pred_cls = np.array([])
    gt_cls = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(cfg.gpu, non_blocking=True)
                labels = labels.cuda(cfg.gpu, non_blocking=True)

            output = model(images)
            loss = torch.mean(criterion(output, labels))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            labels_one_hot = F.one_hot(labels, cfg.num_classes)
            predict_one_hot = F.one_hot(predicted, cfg.num_classes)
            class_num = class_num + labels_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (labels_one_hot + predict_one_hot
                                 == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            cfd_part, pred_cls_part = torch.max(prob, dim=1)
            cfd = np.append(cfd, cfd_part.cpu().numpy())
            pred_cls = np.append(pred_cls, pred_cls_part.cpu().numpy())
            gt_cls = np.append(gt_cls, labels.cpu().numpy())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        acc_cls = correct / class_num
        h_acc = acc_cls[cfg.h_class_idx[0]:cfg.h_class_idx[1]].mean() * 100
        m_acc = acc_cls[cfg.m_class_idx[0]:cfg.m_class_idx[1]].mean() * 100
        t_acc = acc_cls[cfg.t_class_idx[0]:cfg.t_class_idx[1]].mean() * 100
        cal = calibration(gt_cls, pred_cls, cfd, num_bins=15)

        if not cfg.debug:
            logger.info(f'Epoch [{epoch}]:\n')
            logger.info(
                '* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'
                .format(top1=top1,
                        top5=top5,
                        head_acc=h_acc,
                        med_acc=m_acc,
                        tail_acc=t_acc))
            logger.info('* ECE   {ece:.3f}%.'.format(
                ece=cal['expected_calibration_error'] * 100))

    return top1.avg, cal['expected_calibration_error'] * 100
