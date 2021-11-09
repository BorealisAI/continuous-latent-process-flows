# MIT License

# Copyright (c) 2019 Ricky Tian Qi Chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################
# Residual Flows for Invertible Generative Modeling 
# Author: Ricky T. Q. Chen
# Link: https://github.com/rtqichen/residual-flows
###################################################

import torch
import torchvision.datasets as vdsets


class Dataset(object):

    def __init__(self, loc, transform=None, in_mem=True):
        self.in_mem = in_mem
        self.dataset = torch.load(loc)
        if in_mem: self.dataset = self.dataset.float().div(255)
        self.transform = transform

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        x = self.dataset[index]
        if not self.in_mem: x = x.float().div(255)
        x = self.transform(x) if self.transform is not None else x
        return x, 0


class MNIST(object):

    def __init__(self, dataroot, train=True, transform=None):
        self.mnist = vdsets.MNIST(dataroot, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.mnist)

    @property
    def ndim(self):
        return 1

    def __getitem__(self, index):
        return self.mnist[index]


class CIFAR10(object):

    def __init__(self, dataroot, train=True, transform=None):
        self.cifar10 = vdsets.CIFAR10(dataroot, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.cifar10)

    @property
    def ndim(self):
        return 3

    def __getitem__(self, index):
        return self.cifar10[index]


class CelebA5bit(object):

    LOC = 'data/celebahq64_5bit/celeba_full_64x64_5bit.pth'

    def __init__(self, train=True, transform=None):
        self.dataset = torch.load(self.LOC).float().div(31)
        if not train:
            self.dataset = self.dataset[:5000]
        self.transform = transform

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        x = self.dataset[index]
        x = self.transform(x) if self.transform is not None else x
        return x, 0


class CelebAHQ(Dataset):
    TRAIN_LOC = 'data/celebahq/celeba256_train.pth'
    TEST_LOC = 'data/celebahq/celeba256_validation.pth'

    def __init__(self, train=True, transform=None):
        return super(CelebAHQ, self).__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform)


class Imagenet32(Dataset):
    TRAIN_LOC = 'data/imagenet32/train_32x32.pth'
    TEST_LOC = 'data/imagenet32/valid_32x32.pth'

    def __init__(self, train=True, transform=None):
        return super(Imagenet32, self).__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform)


class Imagenet64(Dataset):
    TRAIN_LOC = 'data/imagenet64/train_64x64.pth'
    TEST_LOC = 'data/imagenet64/valid_64x64.pth'

    def __init__(self, train=True, transform=None):
        return super(Imagenet64, self).__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform, in_mem=False)
