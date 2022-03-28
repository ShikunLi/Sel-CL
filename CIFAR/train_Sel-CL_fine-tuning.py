import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
import argparse
import os
import time

from dataset.cifar_dataset import *
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models

import random
import sys

sys.path.append('../utils')
from test_eval import test_eval
from utils_plus import *
from other_utils import *
from models.preact_resnet import *

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    parser.add_argument('--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of in-distribution classes')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--noise_type', default='symmetric', help='noise type of the dataset')
    parser.add_argument('--train_root', default='./dataset', help='root for train data')
    parser.add_argument('--noise_ratio', type=float, default=0.5, help='percent of noise')
    parser.add_argument('--out', type=str, default='./out', help='Directory of the output')
    parser.add_argument('--alpha_m', type=float, default=1.0, help='Beta distribution parameter for mixup')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')
    parser.add_argument('--network', type=str, default='PR18', help='Network architecture')
    parser.add_argument('--seed_initialization', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='name of the experiment (for the output files)')
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='CIFAR-10, CIFAR-100')
    parser.add_argument('--initial_epoch', type=int, default=1, help="Star training at initial_epoch")
    parser.add_argument('--low_dim', type=int, default=128, help='Size of contrastive learning embedding')
    parser.add_argument('--headType', type=str, default="Linear", help='Linear, NonLinear')
    parser.add_argument('--startLabelCorrection', type=int, default=30, help='Epoch to start label correction')
    parser.add_argument('--ReInitializeClassif', type=int, default=0, help='Enable predictive label correction')
    parser.add_argument('--DA', type=str, default="simple", help='Choose simple or complex data augmentation')
    
    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test, clean_idx):

    trainset, testset  = get_dataset(args, TwoTransform(transform_train,transform_test), transform_test)

    ## Get only detected clean samples
    trainset.data = trainset.data[clean_idx==1]
    trainset.targets = trainset.targets[clean_idx==1]

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print('############# Data loaded #############')

    return train_loader, test_loader, trainset



def main(args):
    # best_ac only record the best top1_ac for validation set.
    best_ac = 0.0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed

    random.seed(args.seed_initialization)  # python seed for image transformation

    if args.dataset == 'CIFAR-10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'CIFAR-100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

    if args.DA == "complex":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    exp_path = os.path.join(args.out, 'noise_models_' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name, args.seed_initialization, args.seed_dataset), args.noise_type, str(args.noise_ratio))
    res_path = os.path.join(args.out, 'metrics' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name, args.seed_initialization, args.seed_dataset),  args.noise_type, str(args.noise_ratio))

    model = PreActResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    
    try:
        model.load_state_dict(torch.load(exp_path + "/Sel-CL_model.pth")['model'],strict=False)
    except:
        model.load_state_dict(torch.load(exp_path + "/Sel-CL_model.pth"),strict=False)
        
    clean_idx = np.load( res_path + "/selected_examples_train.npy")

    train_loader, test_loader, trainset = data_config(args, transform_train, transform_test, clean_idx)

    exp_path = exp_path + "/plus/"
    res_path = res_path + "/plus/"
    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    __console__=sys.stdout
    log_file=open(res_path+"results.log",'a')
    sys.stdout=log_file
    print(args)
    
    if args.ReInitializeClassif==1:
        model.linear2 = nn.Linear(512, args.num_classes).to(device)

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    for epoch in range(args.initial_epoch, args.epoch + 1):
        st = time.time()
        print("=================>    ", args.experiment_name, args.noise_ratio)
        scheduler.step()
        train_mixup(args, model, device, train_loader, optimizer, epoch)
        print('Epoch time: {:.2f} seconds\n'.format(time.time()-st))

        test_eval(args, model, device, test_loader)
    save_model(model, optimizer, args, epoch, exp_path+"/Sel-CL_plus_model.pth")


if __name__ == "__main__":
    args = parse_args()
    main(args)
