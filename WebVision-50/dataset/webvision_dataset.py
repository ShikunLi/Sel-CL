import os
import pickle
import torchvision as tv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import time
from IPython import embed

def sample_traning_set(train_imgs, labels, num_class, num_samples):
    random.shuffle(train_imgs)
    class_num = torch.zeros(num_class)
    sampled_train_imgs = []
    for impath in train_imgs:
        label = labels[impath]
        if class_num[label] < (num_samples / num_class):
            sampled_train_imgs.append(impath)
            class_num[label] += 1
        if len(sampled_train_imgs) >= num_samples:
            break
    return sampled_train_imgs
    
def get_dataset(args, transform_train, transform_test):
    train_dataset = webvision_dataset(root_dir=args.trainval_root, transform=transform_train, target_transform=transform_test, mode='all', num_class=50)

    #################################### Test set #############################################
    test_dataset_1 = webvision_dataset(root_dir=args.trainval_root, transform=transform_test, target_transform=transform_test, mode='test', num_class=50)
    
    test_dataset_2 = imagenet_dataset(root_dir=args.val_root, web_root=args.trainval_root, transform=transform_test,  num_class=50)
        
    return train_dataset, test_dataset_1, test_dataset_2

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, web_root, transform, num_class):
        self.root = root_dir
        self.transform = transform
        self.val_data = []
        with open(os.path.join(web_root, 'info/synsets.txt')) as f:
            lines = f.readlines()
        synsets = [x.split()[0] for x in lines]
        for c in range(num_class):
            class_path = os.path.join(self.root, synsets[c])
            imgs = os.listdir(class_path)
            for img in imgs:
                self.val_data.append([c, os.path.join(class_path, img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)
        
class webvision_dataset(Dataset):
    def __init__(self, root_dir, transform, target_transform, mode, num_class, num_samples=None, pred=[], probability=[], paths=[],
                 log=''):
        self.root = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if self.mode == 'test':
            with open(os.path.join(self.root, 'info/val_filelist.txt')) as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
        else:
            with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
                lines = f.readlines()
            if num_class == 1000:
                with open(os.path.join(self.root, 'info/train_filelist_flickr.txt')) as f:
                    lines += f.readlines()
            train_imgs = []
            self.train_labels = []
            self.targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    self.train_labels.append(target)
                    self.targets.append(target)
            self.targets=np.array(self.targets)
            if self.mode == 'all':
                if num_samples is not None:
                    self.train_imgs = sample_traning_set(train_imgs, self.train_labels, num_class, num_samples)
                else:
                    self.train_imgs = train_imgs
            self.train_imgs = np.array(self.train_imgs)
            self.train_labels = np.array(self.train_labels)
    def __getitem__(self, index):
        if  self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img1 = self.transform(image)
            return img1, target, index
        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(os.path.join(self.root, 'val_images_256/', img_path)).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)