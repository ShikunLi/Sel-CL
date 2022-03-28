import os
import pickle
import torchvision as tv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import time
from IPython import embed
import json
import time

def get_dataset(args, transform_train, transform_test):
    # prepare datasets
    #################################### Train set #############################################
    if(args.dataset=='CIFAR-10'):
        cifar_train = Cifar10Train(args, train=True, transform=transform_train, target_transform=transform_test, download=args.download)
    else:
        cifar_train = Cifar100Train(args, train=True, transform=transform_train, target_transform=transform_test, download=args.download)
    #################################### Noise corruption ######################################
    
    if args.noise_type == "symmetric":
        cifar_train.random_in_noise()

    elif args.noise_type == "asymmetric":
        cifar_train.real_in_noise()

    else:
        print('No noise')
    cifar_train.labelsNoisyOriginal = cifar_train.targets.copy()

    #################################### Test set #############################################
    if(args.dataset=='CIFAR-10'):
        testset = tv.datasets.CIFAR10(root=args.train_root, train=False, download=args.download, transform=transform_test)
    else:
        testset = tv.datasets.CIFAR100(root=args.train_root, train=False, download=args.download, transform=transform_test)
    ###########################################################################################

    return cifar_train, testset


class Cifar10Train(tv.datasets.CIFAR10):
    def __init__(self, args, train=True, transform=None, target_transform=None, sample_indexes=None, download=False):
        super(Cifar10Train, self).__init__(args.train_root, train=train, transform=transform,
                                            target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])

        self.num_classes = self.args.num_classes
        self.in_index = []
        self.out_index = []
        self.noisy_indexes = []
        self.clean_indexes = []
        self.clean_labels = []
        self.noisy_labels = []
        self.out_data = []
        self.out_labels = []
        self.soft_labels = []
        self.labelsNoisyOriginal = []
        self._num = []
        self._count = 1
        self.prediction = []
        self.confusion_matrix_in = np.array([])
        self.confusion_matrix_out = np.array([])
        self.labeled_idx = []
        self.unlabeled_idx = []


        # From in ou split function:
        self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
        self._num = int(len(self.targets) * self.args.noise_ratio)
        
    ################# Random in-distribution noise #########################
    def random_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)
        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 0 ## Remove soft-label created during label mapping
                # targets[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                # while(label_sym==self.targets[idxes[i]]):#To exclude the original label
                    # label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.targets[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

        self.targets = np.asarray(self.targets, dtype=np.long)
        self.noisy_labels = np.copy(self.targets)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
        self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(self.args.num_classes))\
                                    *(self.args.noise_ratio/(self.num_classes -1)) + \
                                    np.identity(self.args.num_classes)*(1 - self.args.noise_ratio)
        print('clean_num',sum(self.noisy_labels==self.clean_labels))

    ##########################################################################


    ################# Real in-distribution noise #########################

    def real_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        ##### Create te confusion matrix #####

        self.confusion_matrix_in = np.identity(self.args.num_classes)

        # truck -> automobile
        self.confusion_matrix_in[9, 9] = 1 - self.args.noise_ratio
        self.confusion_matrix_in[9, 1] = self.args.noise_ratio

        # bird -> airplane
        self.confusion_matrix_in[2, 2] = 1 - self.args.noise_ratio
        self.confusion_matrix_in[2, 0] = self.args.noise_ratio

        # cat -> dog
        self.confusion_matrix_in[3, 3] = 1 - self.args.noise_ratio
        self.confusion_matrix_in[3, 5] = self.args.noise_ratio

        # dog -> cat
        self.confusion_matrix_in[5, 5] = 1 - self.args.noise_ratio
        self.confusion_matrix_in[5, 3] = self.args.noise_ratio

        # deer -> horse
        self.confusion_matrix_in[4, 4] = 1 - self.args.noise_ratio
        self.confusion_matrix_in[4, 7] = self.args.noise_ratio

        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)

        for i in range(len(idxes)):
            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 0  ## Remove soft-label created during label mapping
            current_label = self.targets[idxes[i]]
            if self._num > 0:
                # current_label = self.targets[idxes[i]]
                conf_vec = self.confusion_matrix_in[current_label,:]
                label_sym = np.random.choice(np.arange(0, self.num_classes), p=conf_vec.transpose())
                self.targets[idxes[i]] = label_sym
            else:
                label_sym = current_label

            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

            if label_sym == current_label:
                self.clean_indexes.append(idxes[i])
            else:
                self.noisy_indexes.append(idxes[i])

        self.targets = np.asarray(self.targets, dtype=np.long)
        self.clean_indexes = np.asarray(self.clean_indexes, dtype=np.long)
        self.noisy_indexes = np.asarray(self.noisy_indexes, dtype=np.long)
        self.noisy_labels = self.targets
        self.clean_labels = clean_labels
        print('clean_num',sum(self.noisy_labels==self.clean_labels))

    def __getitem__(self, index):
        if self.train:
            img, labels = self.data[index], self.targets[index]
            
            img = Image.fromarray(img)

            img1 = self.transform(img)
            
            return img1, labels, index

        else:
            img, labels = self.data[index], self.targets[index]
            # doing this so that it is consistent with all other datasets.
            img = Image.fromarray(img)

            img = self.transform(img)

            return img, labels
            
            
class Cifar100Train(tv.datasets.CIFAR100):
    def __init__(self, args, train=True, transform=None, target_transform=None, sample_indexes=None, download=False):
        super(Cifar100Train, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])

        self.num_classes = self.args.num_classes
        self.in_index = []
        self.out_index = []
        self.noisy_indexes = []
        self.clean_indexes = []
        self.clean_labels = []
        self.noisy_labels = []
        self.out_data = []
        self.out_labels = []
        self.soft_labels = []
        self.labelsNoisyOriginal = []
        self._num = []
        self._count = 1
        self.prediction = []
        self.confusion_matrix_in = np.array([])
        self.confusion_matrix_out = np.array([])
        self.labeled_idx = []
        self.unlabeled_idx = []


        # From in ou split function:
        self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
        self._num = int(len(self.targets) * self.args.noise_ratio)


    def random_in_noise(self):

        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)
        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 0 ## Remove soft-label created during label mapping
                # train_labels[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                # while(label_sym==self.targets[idxes[i]]):#To exclude the original label
                    # label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.targets[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

        self.targets = np.asarray(self.targets, dtype=np.long)
        self.noisy_labels = np.copy(self.targets)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
        self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(self.args.num_classes))\
                                    *(self.args.noise_ratio/(self.num_classes -1)) + \
                                    np.identity(self.args.num_classes)*(1 - self.args.noise_ratio)
        print('clean_num',sum(self.noisy_labels==self.clean_labels))


    ##########################################################################


    ################# Asymmetric noise #########################

    def real_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        ##### Create te confusion matrix #####

        self.confusion_matrix_in = np.identity(self.args.num_classes) * (1 - self.args.noise_ratio)

        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)

        with open(self.root +'/cifar-100-python/train', 'rb') as f:
            entry = pickle.load(f, encoding='latin1')

        coarse_targets = np.asarray(entry['coarse_labels'])


        targets = np.array(self.targets)
        num_subclasses = self.args.num_classes // 20

        for i in range(20):
            # embed()
            subclass_targets = np.unique(targets[coarse_targets == i])
            clean = subclass_targets
            noisy = np.concatenate([clean[1:], clean[:1]])
            for j in range(num_subclasses):
                self.confusion_matrix_in[clean[j], noisy[j]] = self.args.noise_ratio



        for t in range(len(idxes)):
            self.soft_labels[idxes[t]][self.targets[idxes[t]]] = 0  ## Remove soft-label created during label mapping
            current_label = self.targets[idxes[t]]
            conf_vec = self.confusion_matrix_in[current_label,:]
            label_sym = np.random.choice(np.arange(0, self.num_classes), p=conf_vec.transpose())
            self.targets[idxes[t]] = label_sym
            self.soft_labels[idxes[t]][self.targets[idxes[t]]] = 1

            if label_sym == current_label:
                self.clean_indexes.append(idxes[t])
            else:
                self.noisy_indexes.append(idxes[t])

        self.targets = np.asarray(self.targets, dtype=np.long)
        self.clean_indexes = np.asarray(self.clean_indexes, dtype=np.long)
        self.noisy_indexes = np.asarray(self.noisy_indexes, dtype=np.long)
        self.noisy_labels = self.targets
        self.clean_labels = clean_labels
        print('clean_num',sum(self.noisy_labels==self.clean_labels))

    def __getitem__(self, index):
        if self.train:
            img, labels = self.data[index], self.targets[index]
            
            img = Image.fromarray(img)

            img1 = self.transform(img)
            
            return img1, labels, index

        else:
            img, labels = self.data[index], self.targets[index]
            # doing this so that it is consistent with all other datasets.
            img = Image.fromarray(img)

            img = self.transform(img)

            return img, labels