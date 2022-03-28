
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
import argparse
import os
import time

from dataset.webvision_dataset import *
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models

import random
import sys

sys.path.append('../utils')
from utils_noise_webvision import *
from utils_plus_webvision import *
from test_eval import test_eval
from other_utils import *
import models_webvision as mod


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=50, help='training epoches')
    parser.add_argument('--num_classes', type=int, default=50, help='Number of in-distribution classes')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--trainval_root', default='./dataset/webvision-50/', help='root for train data')
    parser.add_argument('--val_root', default='./dataset/imagenet/', help='root for imagenet val data')
    parser.add_argument('--out', type=str, default='./out/', help='Directory of the output')
    parser.add_argument('--alpha_m', type=float, default=1.0, help='Beta distribution parameter for mixup')
    parser.add_argument('--network', type=str, default='RN18', help='Network architecture')
    parser.add_argument('--seed_initialization', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='name of the experiment (for the output files)')
    parser.add_argument('--dataset', type=str, default='webvision', help='CIFAR-10, CIFAR-100')
    parser.add_argument('--initial_epoch', type=int, default=1, help="Star training at initial_epoch")
    parser.add_argument('--low_dim', type=int, default=128, help='Size of contrastive learning embedding')
    parser.add_argument('--headType', type=str, default="Linear", help='Linear, NonLinear')
    parser.add_argument('--startLabelCorrection', type=int, default=9999, help='Epoch to start label correction')
    parser.add_argument('--ReInitializeClassif', type=int, default=1, help='Enable predictive label correction')
    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test,clean_idx):

    trainset, testset, imagenet_set = get_dataset(args, TwoTransform(transform_train,transform_test), transform_test)
    
    trainset.train_imgs = trainset.train_imgs[clean_idx==1]
    trainset.train_labels = trainset.train_labels[clean_idx==1]    
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    imagenet_test_loader = torch.utils.data.DataLoader(imagenet_set, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print('############# Data loaded #############')

    return train_loader, test_loader, imagenet_test_loader, trainset

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_path = os.path.join(args.out, 'noise_models_' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                                             args.seed_initialization,
                                                                                             args.seed_dataset))
    res_path = os.path.join(args.out, 'metrics' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                                       args.seed_initialization,
                                                                                       args.seed_dataset))

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed

    random.seed(args.seed_initialization)  # python seed for image transformation

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
        
    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])  


    clean_idx = np.load(res_path+ "/selected_examples_train.npy")
    train_loader, test_loader, imagenet_test_loader, trainset= data_config(args, transform_train, transform_test, clean_idx)
    st = time.time()

    
    model = mod.ResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)

    try:
        load_model = torch.load(exp_path+ "/Sel-CL_model.pth")
    except:
        load_model = torch.load(exp_path+ "/Sel-CL_model_130epoch.pth")
        
    try:
        state_dic={k.replace('module.',''):v for k,v in load_model['model'].items()}
    except:
        state_dic={k.replace('module.',''):v for k,v in load_model.items()}
    model.load_state_dict(state_dic)
        

    if args.ReInitializeClassif==1:
        model.linear2 = nn.Linear(512, args.num_classes).to(device)
        
    model = nn.DataParallel(model)

    res_path = res_path +"/plus/"
    exp_path = exp_path +"/plus/"
    

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)


    __console__=sys.stdout
    name= "/results"
    log_file=open(res_path+name+".log",'a')
    sys.stdout=log_file
    print(args)

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_acc_val_1 =0
    best_acc_val_2 =0
    for epoch in range(args.initial_epoch, args.epoch + 1):
        st = time.time()
        print("=================>    ", args.experiment_name)

        scheduler.step()
        train_mixup(args, model, device, train_loader, optimizer, epoch)

        _, acc_val_1 = test_eval(args, model, device, test_loader)
        _, acc_val_2 = test_eval(args, model, device, imagenet_test_loader)


        print('Epoch time: {:.2f} seconds\n'.format(time.time()-st))

        if epoch == args.initial_epoch:
            best_acc_val_1 = acc_val_1
            best_acc_val_2 = acc_val_2
        else:
            if acc_val_1 > best_acc_val_1:
                torch.save(model.state_dict(), os.path.join(exp_path, 'best_val_sel-cl_model.pth'))
                best_acc_val_1 = acc_val_1
            if acc_val_2 > best_acc_val_2:
                best_acc_val_2 = acc_val_2

    print('Best acc:', best_acc_val_1, best_acc_val_2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
