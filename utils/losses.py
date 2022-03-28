import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def Supervised_ContrastiveLearning_loss(args, pairwise_comp, maskSup, mask2Sup, maskUnsup, mask2Unsup, logits_mask, lam1, lam2, bsz, epoch, device,batch_idx):

    logits = torch.div(pairwise_comp, args.sup_t)

    exp_logits = torch.exp(logits) * logits_mask  # remove diagonal

    if args.aprox == 1:
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob = torch.log(torch.exp(logits) + 1e-7) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

    exp_logits2 = torch.exp(logits) * logits_mask  # remove diagonal

    if args.aprox == 1:
        log_prob2 = logits - torch.log(exp_logits2.sum(1, keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob2 = torch.log(torch.exp(logits) + 1e-7) - torch.log(exp_logits2.sum(1, keepdim=True) + 1e-7)

    # compute mean of log-likelihood over positive (weight individual loss terms with mixing coefficients)
    mean_log_prob_pos_sup = (maskSup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))
    mean_log_prob_pos_unsup = (maskUnsup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))
    ## Second mixup term log-probs
    mean_log_prob_pos2_sup = (mask2Sup * log_prob2).sum(1) / (mask2Sup.sum(1) + mask2Unsup.sum(1))
    mean_log_prob_pos2_unsup = (mask2Unsup * log_prob2).sum(1) / (mask2Sup.sum(1) + mask2Unsup.sum(1))

    ## Weight first and second mixup term (both data views) with the corresponding mixing weight

    ##First mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss1a = -lam1 * mean_log_prob_pos_unsup[:int(len(mean_log_prob_pos_unsup) / 2)] - lam1 * mean_log_prob_pos_sup[:int(len(mean_log_prob_pos_sup) / 2)]
    ##First mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss1b = -lam2 * mean_log_prob_pos_unsup[int(len(mean_log_prob_pos_unsup) / 2):] - lam2 * mean_log_prob_pos_sup[int(len(mean_log_prob_pos_sup) / 2):]
    ## All losses for first mixup term
    loss1 = torch.cat((loss1a, loss1b))

    ##Second mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss2a = -(1.0 - lam1) * mean_log_prob_pos2_unsup[:int(len(mean_log_prob_pos2_unsup) / 2)] - (1.0 - lam1) * mean_log_prob_pos2_sup[:int(len(mean_log_prob_pos2_sup) / 2)]
    ##Second mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss2b = -(1.0 - lam2) * mean_log_prob_pos2_unsup[int(len(mean_log_prob_pos2_unsup) / 2):] - (1.0 - lam2) * mean_log_prob_pos2_sup[int(len(mean_log_prob_pos2_sup) / 2):]
    ## All losses secondfor first mixup term
    loss2 = torch.cat((loss2a, loss2b))

    ## Final loss (summation of mixup terms after weighting)
    loss = loss1 + loss2

    loss = loss.view(2, bsz).mean(dim=0)
    
    loss = ((maskSup[:bsz].sum(1))>0)*(loss.view(bsz))
    return loss.mean()


def ClassificationLoss(args, predsA, predsB, y_a1, y_b1, y_a2, y_b2, mix_index1, mix_index2, lam1, lam2, criterionCE, epoch, device):

    preds = torch.cat((predsA, predsB), dim=0)

    targets_1 = torch.cat((y_a1, y_a2), dim=0)
    targets_2 = torch.cat((y_b1, y_b2), dim=0)
    mix_index = torch.cat((mix_index1, mix_index2), dim=0)

    ones_vec = torch.ones((predsA.size()[0],)).float().to(device)
    lam_vec = torch.cat((lam1 * ones_vec, lam2 * ones_vec), dim=0).to(device)

    loss = lam_vec * criterionCE(preds, targets_1) + (1 - lam_vec) * criterionCE(preds, targets_2)
    loss = loss.mean()
    return loss

def Simi_loss(args, pros_simi, maskSup, mask2Sup, maskUnsup, mask2Unsup, logits_mask, lam1, lam2, bsz, epoch, device,batch_idx):

    loss_simi_1 = -(maskSup+maskUnsup)*torch.log(pros_simi+1e-7)-(1-maskSup-maskUnsup)*torch.log(1-pros_simi+1e-7)
    loss_simi_1 =  logits_mask*loss_simi_1
    loss_simi_2 = -(mask2Sup+mask2Unsup)*torch.log(pros_simi+1e-7)-(1-mask2Sup-mask2Unsup)*torch.log(1-pros_simi+1e-7)
    loss_simi_2 =  logits_mask*loss_simi_2

    loss1a = lam1 * loss_simi_1[:int(len(loss_simi_1) / 2)]
    loss1b = lam2 * loss_simi_1[int(len(loss_simi_1) / 2):]
    loss2a = ( 1.0 - lam1) * loss_simi_2[:int(len(loss_simi_2) / 2)]
    loss2b = ( 1.0 - lam2) * loss_simi_2[int(len(loss_simi_2) / 2):]
    
    loss_simi = torch.cat((loss1a+loss2a, loss1b+loss2b)).mean()

    return loss_simi
