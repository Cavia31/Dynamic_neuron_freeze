import models.mobile_net_v2 as mbv2
import models.simple_models as smod

import torch
from torch.utils.data import random_split,DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as T


class Builder():

    def __init__(self, model_config, dataset_config):
        
        for attr_name in model_config:
            # set model attributes
            self.__setattr__(attr_name, model_config[attr_name])

        for attr_name in dataset_config:
            # set dataset attributes
            self.__setattr__(attr_name, dataset_config[attr_name])

        #Initialize the dataset
        self.dataset_name = self.dataset
        if self.dataset == 'mnist':
            train_transform = T.Compose([
                T.Normalize(0, 1)
            ])
            test_transform = train_transform
            self.dataset = {
                'train': datasets.MNIST("./datasets/MNIST", train=True, transform=train_transform, download=True),
                'test' : datasets.MNIST("./datasets/MNIST", train=False, transform=test_transform, download=True)
                }
        elif self.dataset == 'c10':
            train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(32, 4),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
            test_transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
            self.dataset = {
                'train': datasets.CIFAR10("./datasets/CIFAR10", train=True, transform=train_transform, download=True),
                'test': datasets.CIFAR10("./datasets/CIFAR10", train=False, transform=test_transform, download=True)
            }
        elif self.dataset == 'c100':
            train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(32, 4),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            test_transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.dataset = {
                'train': datasets.CIFAR100("./datasets/CIFAR100", train=True, transform=train_transform, download=True),
                'test': datasets.CIFAR100("./datasets/CIFAR100", train=False, transform=test_transform, download=True)
            }
        
        if self.valid:
            train_len = int(len(self.dataset['train'])) - self.valid_len
            self.dataset['train'], self.dataset['valid'] = random_split(
                self.dataset['train'],
                [train_len, self.valid_len],
                generator=torch.Generator().manual_seed(self.seed),
                )
            
        self.dataloader = {
            'train': DataLoader(dataset=self.dataset['train'], batch_size=self.batch_size, shuffle=True),
            'test': DataLoader(dataset=self.dataset['test'], batch_size=self.batch_size, shuffle=False)
        }
        if self.valid:
            self.dataloader['valid'] = DataLoader(dataset=self.dataset['valid'], batch_size=self.batch_size, shuffle=True)
        
        self.n_classes = len(self.dataset['train'].classes)
        self.classes = self.dataset['train'].classes


        # Initialize the model
        self.model_name = self.model
        if self.model == "AlexNet":
            self.model = smod.AlexNet(self.n_classes)
        elif self.model == "mbv2":
            self.model = mbv2.freezablemobilenet_v2(
                weights=mbv2.MobileNet_V2_Weights.IMAGENET1K_V2 if self.weights == 'pretrained' else None,
                progress = True,
                num_classes=self.n_classes,
                inverted_residual_setting=self.dataset_name
                )

#b = Builder({'model': "A"}, {'dataset': "c100"})