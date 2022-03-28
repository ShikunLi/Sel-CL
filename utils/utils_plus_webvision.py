from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from AverageMeter import AverageMeter
from criterion import *
from utils_noise import *
import time
import warnings
import os, sys
import torch.cuda.amp as amp
warnings.filterwarnings('ignore')

def train_mixup(args, model, device, train_loader, optimizer, epoch):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    counter = 1

    criterionCE = torch.nn.CrossEntropyLoss(reduction="none")
    scaler = amp.GradScaler()
    for batch_idx, (img, labels, index) in enumerate(train_loader):

        img1, img_noDA, labels, index = img[0].to(device), img[1].to(device), labels.to(device), index.to(device)

        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha_m, device)

        model.zero_grad()

        with amp.autocast():
            predsA, _ = model(img1)

            ## Forward pass free of DA
            predsNoDA, _ = model(img_noDA)
            predsNoDA = predsNoDA.detach()

            ## Compute classification loss (returned individual per-sample loss)
            lossClassif = criterionMixBoot(args, predsA, predsNoDA, y_a1, y_b1, mix_index1, lam1, criterionCE, epoch,
                                           device)

            ## Average loss after saving it per-sample
            loss = lossClassif.mean()

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        #scaler.unscale_(optimizer)
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2) 
        scaler.step(optimizer)
        scaler.update()

        prec1, prec5 = accuracy_v2(predsNoDA, labels, top=[1, 5])
        train_loss.update(loss.item(), img1.size(0))
        top1.update(prec1.item(), img1.size(0))
        top5.update(prec5.item(), img1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                optimizer.param_groups[0]['lr']))
        counter = counter + 1

    return train_loss.avg, top1.avg, top5.avg, batch_time.sum

def criterionMixBoot(args, preds, predsNoDA, targets_1, targets_2, mix_index, lam, criterionCE, epoch, device):
    lam_vec = lam * torch.ones((preds.size()[0],)).float().to(device)

    if epoch <= args.startLabelCorrection:
        loss = lam_vec * criterionCE(preds, targets_1) + (1 - lam_vec) * criterionCE(preds, targets_2)
    
    else:
        output_x1 = F.log_softmax(predsNoDA, dim=1)
        output_x2 = output_x1[mix_index, :]
        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        B = 0.2

        loss_x1_vec = lam_vec * (1 - B) * criterionCE(preds, targets_1)
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)

        loss_x1_pred_vec = lam_vec * B * criterionCE(preds, z1)
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)

        loss_x2_vec = (1 - lam_vec) * (1 - B) * criterionCE(preds, targets_2)
        loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)

        loss_x2_pred_vec = (1 - lam_vec) * B * criterionCE(preds, z2)
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        loss = loss_x1 + loss_x1_pred + loss_x2 + loss_x2_pred

    return loss