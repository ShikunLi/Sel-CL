from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from AverageMeter import AverageMeter
from NCECriterion import NCESoftmaxLoss
from other_utils import set_bn_train, moment_update
from criterion import *
from utils_mixup import *
from losses import *
import time
import warnings
import os, sys
import torch.cuda.amp as amp
warnings.filterwarnings('ignore')

def train_sel(args, scheduler,model,model_ema,contrast,queue,device, train_loader, train_selected_loader, optimizer, epoch,selected_pairs,log_file):
    train_loss_1 = AverageMeter()
    train_loss_2 = AverageMeter()
    train_loss_3 = AverageMeter()      

    # switch to train mode
    model.train()
    set_bn_train(model_ema)
    end = time.time()
    counter = 1
    
    scaler = amp.GradScaler()
    criterionCE = torch.nn.CrossEntropyLoss(reduction="none")
    criterion = NCESoftmaxLoss(reduction="none").cuda()
    train_selected_loader_iter = iter(train_selected_loader)
    for batch_idx, (img, labels, index) in enumerate(train_loader):

        img1, img2, labels, index = img[0].to(device), img[1].to(device), labels.to(device), index.to(device)

        bsz = img1.shape[0]
        
        model.zero_grad()

        with amp.autocast():
            ##compute uns-cl loss
            _,feat_q = model(img1)

            with torch.no_grad():
                _, feat_k= model_ema(img2)

            out = contrast(feat_q, feat_k, feat_k, update=True)
            uns_loss = criterion(out)          
            
            ##compute sup-cl loss with selected pairs (adapted from MOIT)
            img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha_m, device)
            img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, args.alpha_m, device)

            predsA, embedA = model(img1)
            predsB, embedB = model(img2)
            predsA = F.softmax(predsA,-1)
            predsB = F.softmax(predsB,-1)
            
            with torch.no_grad():
                predsA_ema, embedA_ema = model_ema(img1)
                predsB_ema, embedB_ema = model_ema(img2)
                predsA_ema = F.softmax(predsA_ema,-1)
                predsB_ema = F.softmax(predsB_ema,-1)

            
            if args.sup_queue_use == 1:
                queue.enqueue_dequeue(torch.cat((embedA_ema.detach(), embedB_ema.detach()), dim=0), torch.cat((predsA_ema.detach(), predsB_ema.detach()), dim=0), torch.cat((index.detach().squeeze(), index.detach().squeeze()), dim=0))

            if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:
                queue_feats, queue_pros, queue_index = queue.get()
                    
            else:
                queue_feats, queue_pros, queue_index = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
            

            maskUnsup_batch, maskUnsup_mem, mask2Unsup_batch, mask2Unsup_mem = unsupervised_masks_estimation(args, queue, mix_index1, mix_index2, epoch, bsz, device)

            embeds_batch = torch.cat([embedA, embedB], dim=0)
            pros_batch = torch.cat([predsA, predsB], dim=0)
            pairwise_comp_batch = torch.matmul(embeds_batch, embeds_batch.t())
            pros_simi_batch = torch.mm(pros_batch,pros_batch.t())

            if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:
                embeds_mem = torch.cat([embedA, embedB, queue_feats], dim=0)
                pros_mem = torch.cat([predsA, predsB, queue_pros], dim=0)
                pairwise_comp_mem = torch.matmul(embeds_mem[:2 * bsz], embeds_mem[2 * bsz:].t()) ##Compare mini-batch with memory
                pros_simi_mem = torch.mm(pros_mem[:2 * bsz],pros_mem[2 * bsz:].t())

            maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem = \
                supervised_masks_estimation(args, index.long(), queue, queue_index.long(), mix_index1, mix_index2, epoch, bsz, device,selected_pairs)

            logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(2 * bsz).to(device))  ## Negatives mask, i.e. all except self-contrast sample

            loss_sup = Supervised_ContrastiveLearning_loss(args, pairwise_comp_batch, maskSup_batch, mask2Sup_batch, maskUnsup_batch, mask2Unsup_batch, logits_mask_batch, lam1, lam2, bsz, epoch, device,batch_idx)

            ## compute simi_loss
            loss_simi = Simi_loss(args, pros_simi_batch, maskSup_batch, mask2Sup_batch, maskUnsup_batch, mask2Unsup_batch, logits_mask_batch, lam1, lam2, bsz, epoch, device,batch_idx)
            
            ## using queue
            if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:

                logits_mask_mem = torch.ones_like(maskSup_mem) ## Negatives mask, i.e. all except self-contrast sample

                if queue.ptr == 0:
                    logits_mask_mem[:, -2 * bsz:] = logits_mask_batch
                else:
                    logits_mask_mem[:, queue.ptr - (2 * bsz):queue.ptr] = logits_mask_batch

                loss_mem = Supervised_ContrastiveLearning_loss(args, pairwise_comp_mem, maskSup_mem, mask2Sup_mem, maskUnsup_mem, mask2Unsup_mem, logits_mask_mem, lam1, lam2, bsz, epoch, device,batch_idx)

                loss_sup = loss_sup + loss_mem
                
                loss_simi_mem = Simi_loss(args, pros_simi_mem, maskSup_mem, mask2Sup_mem, maskUnsup_mem, mask2Unsup_mem, logits_mask_mem, lam1, lam2, bsz, epoch, device,batch_idx)
                loss_simi = loss_simi + loss_simi_mem
                
                sel_mask=(maskSup_batch[:bsz].sum(1)+maskSup_mem[:bsz].sum(1))<2
            else:
                sel_mask=(maskSup_batch[:bsz].sum(1))<1
        
        ## compute class loss with selected examples
        try:
            img, labels, _  = next(train_selected_loader_iter)
        except StopIteration:
            train_selected_loader_iter = iter(train_selected_loader)
            img, labels, _ = next(train_selected_loader_iter)
        img1, img2,  labels = img[0].to(device), img[1].to(device), labels.to(device)
        
        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha_m, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, args.alpha_m, device)

        
        with amp.autocast():
            predsA, embedA = model(img1)
            predsB, embedB = model(img2)

            lossClassif = ClassificationLoss(args, predsA, predsB, y_a1, y_b1, y_a2, y_b2, mix_index1,
                                                mix_index2, lam1, lam2, criterionCE, epoch, device)
            
        
        ## compute sel_loss by combining uns-cl loss and  sup-cl loss  
            sel_loss = (sel_mask*uns_loss).mean() + loss_sup
            
            loss = sel_loss + args.lambda_c*lossClassif + args.lambda_s*loss_simi

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2) 
        scaler.step(optimizer)
        scaler.update()
        moment_update(model, model_ema, args.alpha_moving)
        scheduler.step()        
        
        train_loss_1.update(loss_sup.item(), img1.size(0))
        train_loss_2.update(loss_simi.item(), img1.size(0))
        train_loss_3.update(lossClassif.item(), img1.size(0))        

        if counter % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), 0,
                optimizer.param_groups[0]['lr']))
            log_file.flush()
        counter = counter + 1

    print('train_sel_loss',train_loss_1.avg,'train_simi_loss',train_loss_2.avg,'train_class_loss',train_loss_3.avg)
    print('train time', time.time()-end)

def train_uns(args, scheduler,model,model_ema,contrast,queue,device, train_loader, optimizer, epoch,log_file):
    train_loss_1 = AverageMeter() 
    model.train()
    set_bn_train(model_ema)
    end = time.time()
    counter = 1
    
    scaler = amp.GradScaler()
    criterion = NCESoftmaxLoss(reduction="mean").cuda()
    for batch_idx, (img, labels, index) in enumerate(train_loader):

        img1, img2, labels, index = img[0].to(device), img[1].to(device), labels.to(device), index.to(device)

        bsz = img1.shape[0]
        
        model.zero_grad()
        
        with amp.autocast():
            ##compute uns-cl loss
            _,feat_q = model(img1)
            with torch.no_grad():
                _, feat_k= model_ema(img2)

            out = contrast(feat_q, feat_k, feat_k, update=True)
            uns_loss = criterion(out).mean()

            ## update sup queue
            img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha_m, device)
            img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, args.alpha_m, device)

            with torch.no_grad():
                predsA_ema, embedA_ema = model_ema(img1)
                predsB_ema, embedB_ema = model_ema(img2)
                predsA_ema = F.softmax(predsA_ema,-1)
                predsB_ema = F.softmax(predsB_ema,-1)

            if args.sup_queue_use == 1:
                queue.enqueue_dequeue(torch.cat((embedA_ema.detach(), embedB_ema.detach()), dim=0), torch.cat((predsA_ema.detach(), predsB_ema.detach()), dim=0), torch.cat((index.detach().squeeze(), index.detach().squeeze()), dim=0)) 
                
        scaler.scale(uns_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        moment_update(model, model_ema, args.alpha_moving)
        scheduler.step()
        
        train_loss_1.update(uns_loss.item(), img1.size(0)) 

        if counter % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), 0,
                optimizer.param_groups[0]['lr']))
            log_file.flush()
        counter = counter + 1
    print('train_uns_loss',train_loss_1.avg)
    print('train time', time.time()-end)
    
def train_sup(args, scheduler,model,model_ema,contrast,queue,device, train_loader, train_selected_loader, optimizer, epoch,noisy_pairs,log_file):
    train_loss_1 = AverageMeter()
    train_loss_3 = AverageMeter()      

    # switch to train
    model.train()
    set_bn_train(model_ema)
    end = time.time()
    counter = 1
    
    scaler = amp.GradScaler()
    criterionCE = torch.nn.CrossEntropyLoss(reduction="none").cuda()
    train_selected_loader_iter = iter(train_selected_loader)
    for batch_idx, (img, labels, index) in enumerate(train_loader):

        img1, img2, labels, index = img[0].to(device), img[1].to(device), labels.to(device), index.to(device)

        bsz = img1.shape[0]
        
        model.zero_grad()
        
        with amp.autocast():
            ## update uns queue
            _,feat_q = model(img1)

            with torch.no_grad():
                _, feat_k= model_ema(img2)

            contrast(feat_q, feat_k, feat_k, update=True)
            
            ##compute sup-cl loss with noisy pairs (adapted from MOIT)            
            img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, 0, device)
            img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, 0, device)


            predsA, embedA = model(img1)
            predsB, embedB = model(img2)
            predsA = F.softmax(predsA,-1)
            predsB = F.softmax(predsB,-1)
            
            with torch.no_grad():
                predsA_ema, embedA_ema = model_ema(img1)
                predsB_ema, embedB_ema = model_ema(img2)
                predsA_ema = F.softmax(predsA_ema,-1)
                predsB_ema = F.softmax(predsB_ema,-1)

            if args.sup_queue_use == 1:
                queue.enqueue_dequeue(torch.cat((embedA_ema.detach(), embedB_ema.detach()), dim=0), torch.cat((predsA_ema.detach(), predsB_ema.detach()), dim=0), torch.cat((index.detach().squeeze(), index.detach().squeeze()), dim=0))

            if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:
                queue_feats, queue_pros, queue_index = queue.get()
                    
            else:
                queue_feats, queue_pros, queue_index = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

            maskUnsup_batch, maskUnsup_mem, mask2Unsup_batch, mask2Unsup_mem = unsupervised_masks_estimation(args, queue, mix_index1, mix_index2, epoch, bsz, device)

            embeds_batch = torch.cat([embedA, embedB], dim=0)
            pros_batch = torch.cat([predsA, predsB], dim=0)
            pairwise_comp_batch = torch.matmul(embeds_batch, embeds_batch.t())

            if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:
                embeds_mem = torch.cat([embedA, embedB, queue_feats], dim=0)
                pairwise_comp_mem = torch.matmul(embeds_mem[:2 * bsz], embeds_mem[2 * bsz:].t()) ##Compare mini-batch with memory

            maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem = \
                supervised_masks_estimation(args, index.long(), queue, queue_index.long(), mix_index1, mix_index2, epoch, bsz, device,noisy_pairs)


            logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(2 * bsz).to(device))  ## Negatives mask, i.e. all except self-contrast sample

            loss_sup = Supervised_ContrastiveLearning_loss(args, pairwise_comp_batch, maskSup_batch, mask2Sup_batch, maskUnsup_batch, mask2Unsup_batch, logits_mask_batch, lam1, lam2, bsz, epoch, device,batch_idx)
            
            if args.sup_queue_use == 1 and epoch > args.sup_queue_begin:

                logits_mask_mem = torch.ones_like(maskSup_mem) ## Negatives mask, i.e. all except self-contrast sample

                if queue.ptr == 0:
                    logits_mask_mem[:, -2 * bsz:] = logits_mask_batch
                else:
                    logits_mask_mem[:, queue.ptr - (2 * bsz):queue.ptr] = logits_mask_batch

                loss_mem = Supervised_ContrastiveLearning_loss(args, pairwise_comp_mem, maskSup_mem, mask2Sup_mem, maskUnsup_mem, mask2Unsup_mem, logits_mask_mem, lam1, lam2, bsz, epoch, device,batch_idx)

                loss_sup = loss_sup + loss_mem
                
        ## compute class loss with noisy examples
        try:
            img, labels, _  = next(train_selected_loader_iter)
        except StopIteration:
            train_selected_loader_iter = iter(train_selected_loader)
            img, labels, _ = next(train_selected_loader_iter)
        img1, img2,  labels = img[0].to(device), img[1].to(device), labels.to(device)
        
        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, 0, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, 0, device)

        
        with amp.autocast():
            predsA, embedA = model(img1)
            predsB, embedB = model(img2)

            lossClassif = ClassificationLoss(args, predsA, predsB, y_a1, y_b1, y_a2, y_b2, mix_index1,
                                                mix_index2, lam1, lam2, criterionCE, epoch, device)
            
                 
            loss = loss_sup.mean() + args.lambda_c*lossClassif
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        moment_update(model, model_ema, args.alpha_moving)       
        scheduler.step()

      
        train_loss_1.update(loss_sup.item(), img1.size(0))
        train_loss_3.update(lossClassif.item(), img1.size(0))
        if counter % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), 0,
                optimizer.param_groups[0]['lr']))
            log_file.flush()
        counter = counter + 1
        
    print('train_loss_sup',train_loss_1.avg,'train_class_loss',train_loss_3.avg)
    print('train time', time.time()-end)
    
def pair_selection(args, net, device, trainloader, testloader, epoch):

    net.eval()

    C = args.num_classes

    ## Get train features
    transform_bak = trainloader.dataset.transform
    trainloader.dataset.transform = testloader.dataset.transform
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=8)

    trainFeatures = torch.rand(len(trainloader.dataset), args.low_dim).t().to(device)
    smiliar_graph_all=torch.zeros(len(trainloader.dataset),len(trainloader.dataset))
    
    with torch.no_grad():
        for batch_idx, (inputs, _,_) in enumerate(temploader):
            inputs = inputs.to(device)
            batchSize = inputs.size(0)

            _, features = net(inputs)

            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()

    trainNoisyLabels = torch.LongTensor(temploader.dataset.targets).to(device)
    train_new_labels = torch.LongTensor(temploader.dataset.targets).to(device)


    discrepancy_measure1 = torch.zeros((len(temploader.dataset.targets),)).to(device)
    discrepancy_measure2 = torch.zeros((len(temploader.dataset.targets),)).to(device)

    discrepancy_measure1_pseudo_labels = torch.zeros((len(temploader.dataset.targets),)).to(device)
    discrepancy_measure2_pseudo_labels = torch.zeros((len(temploader.dataset.targets),)).to(device)
    
    agreement_measure = torch.zeros((len(temploader.dataset.targets),))

    ## Weighted k-nn correction
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(args.k_val, C).to(device)

        for batch_idx, (inputs,targets, index) in enumerate(temploader):
            targets = targets.to(device)
            batchSize = inputs.size(0)

            features = trainFeatures.t()[index]

            dist = torch.mm(features, trainFeatures)
            smiliar_graph_all[index] = dist.cpu().detach()
            dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1 ##Self-contrast set to -1
            #dist[torch.arange(dist.size()[0]), index] = -1

            yd, yi = dist.topk(args.k_val, dim=1, largest=True, sorted=True) ## Top-K similar scores and corresponding indexes
            candidates = trainNoisyLabels.view(1, -1).expand(batchSize, -1) ##Replicate the labels per row to select
            retrieval = torch.gather(candidates, 1, yi) ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

            retrieval_one_hot_train.resize_(batchSize * args.k_val, C).zero_()
            ## Generate the K*batchSize one-hot encodings from neighboring labels ("retrieval"), i.e. each row in retrieval
            # (set of neighbouring labels) is turned into a one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = torch.exp(yd.clone().div_(args.sup_t)) ## Apply temperature to scores
            yd_transform[...] = 1.0 ##To avoid using similarities
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

            prob_temp = probs_norm[torch.arange(0, batchSize), targets]
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure1[index] = -torch.log(prob_temp)
            
            sorted_pro, predictions_corrected = probs_norm.sort(1, True)
            
            new_labels = predictions_corrected[:, 0]

            prob_temp = probs_norm[torch.arange(0, batchSize), new_labels]
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure1_pseudo_labels[index] = -torch.log(prob_temp)

            train_new_labels[index] =  new_labels
            
    train_new_labels2 = train_new_labels.clone()
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(args.k_val, C).to(device)

        for batch_idx, (inputs,targets, index) in enumerate(temploader):

            targets = targets.to(device)
            batchSize = inputs.size(0)

            features = trainFeatures.t()[index]

            dist = torch.mm(features, trainFeatures)
            dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  ##Self-contrast set to -1

            yd, yi = dist.topk(args.k_val, dim=1, largest=True, sorted=True)  ## Top-K similar scores and corresponding indexes
            candidates = train_new_labels2.view(1, -1).expand(batchSize, -1)  ##Replicate the labels per row to select
            retrieval = torch.gather(candidates, 1, yi)  ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

            retrieval_one_hot_train.resize_(batchSize * args.k_val, C).zero_()
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = torch.exp(yd.clone().div_(args.sup_t))  ## Apply temperature to scores
            yd_transform[...] = 1.0  ##To avoid using similarities only counts
            probs_corrected = torch.sum(
                torch.mul(retrieval_one_hot_train.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)

            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]


            prob_temp = probs_norm[torch.arange(0, batchSize), targets]
            prob_temp[prob_temp<=1e-2] = 1e-2
            prob_temp[prob_temp > (1-1e-2)] = 1-1e-2

            discrepancy_measure2[index] = -torch.log(prob_temp)

            prob_temp = probs_norm[torch.arange(0, batchSize), train_new_labels[index]]
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure2_pseudo_labels[index] = -torch.log(prob_temp)
            
            agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1]==targets).float().data.cpu()
    
    ## select examples   
    num_clean_per_class = torch.zeros(args.num_classes)
    for i in range(args.num_classes):
        idx_class = temploader.dataset.targets==i
        idx_class = torch.from_numpy(idx_class.astype("float")) == 1.0
        num_clean_per_class[i] = torch.sum(agreement_measure[idx_class])
        
    if(args.alpha==0.5):
        num_samples2select_class = torch.median(num_clean_per_class)
    elif(args.alpha==1.0):
        num_samples2select_class = torch.max(num_clean_per_class)
    elif(args.alpha==0.0):
        num_samples2select_class = torch.min(num_clean_per_class)
    else:
        num_samples2select_class = torch.quantile(num_clean_per_class,args.alpha)
    
    agreement_measure = torch.zeros((len(temploader.dataset.targets),)).to(device)

    for i in range(args.num_classes):
        idx_class = temploader.dataset.targets==i
        samplesPerClass = idx_class.sum()
        idx_class = torch.from_numpy(idx_class.astype("float"))
        idx_class = (idx_class==1.0).nonzero().squeeze()
        discrepancy_class = discrepancy_measure2[idx_class]

        if num_samples2select_class>=samplesPerClass:
            k_corrected = samplesPerClass
        else:
            k_corrected = num_samples2select_class

        top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)[1]

        agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0
    
    selected_examples=agreement_measure
    print('selected examples',sum(selected_examples))
    trainloader.dataset.transform = transform_bak
    ## select pairs 
    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples,as_tuple=True)[0].cpu()
        total_selected_num=len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)
        total_num = len(trainNoisyLabels)
        noisy_pairs=torch.eq(trainNoisyLabels, trainNoisyLabels.t())
        
        selected_pairs = noisy_pairs[index_selected.unsqueeze(1).expand(total_selected_num,total_selected_num),index_selected.unsqueeze(0).expand(total_selected_num,total_selected_num)].clone()
        temp_graph = smiliar_graph_all[index_selected.unsqueeze(1).expand(total_selected_num,total_selected_num),index_selected.unsqueeze(0).expand(total_selected_num,total_selected_num)]         
        selected_th=np.quantile(temp_graph[selected_pairs],args.beta)
        print('selected_th',selected_th)
        temp = torch.zeros(total_num,total_num).type(torch.uint8)
        noisy_pairs = torch.where(smiliar_graph_all<selected_th,temp,noisy_pairs.type(torch.uint8)).type(torch.bool)
        noisy_pairs[index_selected.unsqueeze(1).expand(total_selected_num,total_selected_num),index_selected.unsqueeze(0).expand(total_selected_num,total_selected_num)] = selected_pairs
        final_selected_pairs = noisy_pairs                  
        
    return selected_examples.cuda(),final_selected_pairs.contiguous()