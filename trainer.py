import os
import sys
from collections import Counter

import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from sacred import Experiment
from sacred.observers import MongoObserver

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from model.sam import SAM

import warnings
warnings.filterwarnings('ignore')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

url = 'localhost:27017'
db_name = 'ImageClassification'
ex = Experiment('ImageClassifier')
ex.observers.append(MongoObserver.create(url=url, db_name=db_name))


@ex.config
def cfg():
    seed = 128
    adaptive = True
    batch_size = 128
    depth = 16
    dropout = 0.0
    label_smoothing = 0.1
    lr = 0.1
    momentum = 0.9
    rho = 0.5
    weight_decay = 5e-4
    width_factor = 8
    epochs = 20
    res_postfix = 'test'


class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image


class ImageLoader(object):
    def __init__(self, batch_size):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


@ex.capture
def get_dataset(batch_size):
    return ImageLoader(batch_size)


@ex.capture
def get_model(depth, width_factor, dropout, in_channels=3, labels=10):
    model = WideResNet(depth, width_factor, dropout, in_channels=in_channels, labels=labels)
    model_cfg = {
            'depth': depth,
            'width_factor': width_factor,
            'dropout': dropout,
            'in_channels': in_channels,
            'labels': labels
            }
    return model.to(DEVICE), model_cfg


@ex.capture
def get_optimizer(model, rho, adaptive, lr, momentum, weight_decay):
    base_optimizer = optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho, adaptive,
            lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


@ex.capture
def get_scheduler(optimizer, lr, epochs):
    return StepLR(optimizer, lr, epochs)


class StepLR(object):
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class Trainer(object):
    def __init__(self, epochs, res_postfix):
        
        self.epochs = epochs
        self.postfix = res_postfix
        # cifar 10
        dataset = get_dataset()

        self.train_loader = dataset.train 
        self.valid_loader = dataset.test 
       
        self.model, self.cfg = get_model()
        self.optimizer = get_optimizer(model=self.model)
        self.scheduler = get_scheduler(optimizer=self.optimizer)
        self.criterion = smooth_crossentropy

    def save(self, fn=None):
        if fn is None:
            fn = 'model_{}.dat'.format(self.postfix)

        torch.save({
            'model_state_dict' : self.model.state_dict(),
            'cfg': self.cfg
            }, fn)

    def run_batch(self, batch, train):
        inputs, targets = (b.to(DEVICE) for b in batch)
        outputs = self.model(inputs)
        if train:
            loss = self.criterion(outputs, targets).mean()
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            self.criterion(self.model(inputs), targets).mean().backward()
            self.optimizer.second_step(zero_grad=True)

        else:
            loss = self.criterion(outputs, targets).mean()
        

        loss = loss.item()
        bsz = targets.size(0)
        with torch.no_grad():
            correct = (torch.argmax(outputs.data, 1) == targets).sum().item()

        return loss * bsz, correct, bsz

    def train(self, epoch=None):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_cnt = 0
        tqdm_batch_iterator = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_batch_iterator):
            
            loss, correct, bsz = self.run_batch(batch, train=True)
            
            total_loss += loss
            total_correct += correct
            total_cnt += bsz

            description = "[Epoch: {:3d}][Loss: {:6f}][Accuracy: {:6f}][lr: {:7f}]".format(epoch,
                    total_loss/total_cnt,
                    total_correct/total_cnt,
                    self.optimizer.param_groups[0]['lr'])
            tqdm_batch_iterator.set_description(description)
        status = {'train_loss':total_loss/total_cnt,
                'train_acc':total_correct/total_cnt}
        return status

    def valid(self, epoch=None):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total_cnt = 0
            for batch_idx, batch in enumerate(self.valid_loader):
                
                loss, correct, bsz = self.run_batch(batch, train=False)
                
                total_loss += loss
                total_correct += correct
                total_cnt += bsz
        status = {'valid_loss':total_loss/total_cnt,
                'valid_acc':total_correct/total_cnt}
        return status

    def run(self):
        best_loss = 1e9
        train_report = 'train loss : {:.4f} train acc: {:.4f}'
        val_report = 'valid loss : {:.4f} valid acc: {:.4f}'
        for epoch in range(1, self.epochs + 1):
            epoch_status = self.train(epoch)
            print(train_report.format(
                epoch_status['train_loss'],
                epoch_status['train_acc']
                ))
            ex.log_scalar('train_loss', epoch_status['train_loss'], epoch)
            ex.log_scalar('train_acc', epoch_status['train_acc'], epoch)

            valid_status = self.valid(epoch)
            print(val_report.format(
                valid_status['valid_loss'],
                valid_status['valid_acc']
                ))
            ex.log_scalar('valid_loss', valid_status['valid_loss'], epoch)
            ex.log_scalar('valid_acc', valid_status['valid_acc'], epoch)
        
            self.scheduler(epoch)
            if valid_status['valid_loss'] < best_loss:
                best_loss = valid_status['valid_loss']
                self.save()
        self.save(fn='fin_{}.dat'.format(self.postfix))
    

@ex.capture
def get_trainer(epochs, res_postfix):
    return Trainer(epochs, res_postfix)


@ex.automain
def main():
    trainer = get_trainer()
    trainer.run()
