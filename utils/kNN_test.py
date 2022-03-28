import torch
import time
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import os

    
def kNN(args, epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=False, inverse=True, two_branch=False, fusion=None):
    net.eval()
    total = 0
    testsize = testloader.dataset.__len__()

    with torch.no_grad():
        if recompute_memory:
            transform_bak = trainloader.dataset.transform
            trainloader.dataset.transform = testloader.dataset.transform
            temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4)
            for batch_idx, (inputs, _,_) in enumerate(temploader):
                
                batchSize = inputs.size(0)
                inputs = inputs.cuda()

                _,features = net(inputs)
                if batch_idx == 0:
                    trainFeatures = features.data.t()
                else:
                    trainFeatures = torch.cat((trainFeatures, features.data.t()), 1)
                    
            try:
                trainLabels = torch.LongTensor(temploader.dataset.clean_labels).cuda()
            except:
                trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
            trainloader.dataset.transform = transform_bak
        else:
            trainFeatures = lemniscate.memory.t()
            if hasattr(trainloader.dataset, 'imgs'):
                trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
            else:
                try:
                    trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
                except:
                    trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
        C = trainLabels.max() + 1
        C = C.item()

        top1 = 0.
        top5 = 0.

        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            inputs = inputs.cuda()

            _,features = net(inputs)

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = torch.exp(torch.div(yd.clone(), sigma))
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

    return top1/total, top5/total