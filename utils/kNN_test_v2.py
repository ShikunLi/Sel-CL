import torch
import time
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import os
import faiss

def compute_features(args,net,trainloader,testloader):
    net.eval()
    total = 0
    testsize = testloader.dataset.__len__()

    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
        for batch_idx, (inputs, _,_) in enumerate(temploader):
            
            batchSize = inputs.size(0)
            inputs = inputs.cuda()

            _,features = net(inputs)
            if batch_idx == 0:
                trainFeatures = features.data
            else:
                trainFeatures = torch.cat((trainFeatures, features.data), 0)
                    
    trainloader.dataset.transform = transform_bak
    
    return trainFeatures.cpu()

def kNN(args, model, test_loader, k,temperature,features,epoch, trainloader):             
    index = faiss.IndexFlatIP(features.shape[1])       
    index.add(features.cpu().numpy())
    labels = torch.LongTensor(trainloader.dataset.clean_labels)
    soft_labels = torch.zeros(len(labels), args.num_classes).scatter_(1, labels.view(-1,1), 1) 
    top1 = 0.
    top5 = 0.
    total=0
    with torch.no_grad():
        print('==> weighted Knn Testing...')
        model.eval()        
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(non_blocking=True)
            _, test_feat = model(inputs)                  
            batch_size = inputs.size(0)     
            dist = np.zeros((batch_size, k))
            neighbors = np.zeros((batch_size, k))  
            D,I = index.search(test_feat.data.cpu().numpy(),k)                  
            neighbors = torch.LongTensor(I)
            weights = torch.exp(torch.Tensor(D)/temperature).unsqueeze(-1)           
            score = torch.zeros(batch_size,args.num_classes)   
            for n in range(batch_size):           
                neighbor_labels = soft_labels[neighbors[n]]
                score[n] = (neighbor_labels*weights[n]).sum(0)
                
            _, predictions = score.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

    return top1/total, top5/total