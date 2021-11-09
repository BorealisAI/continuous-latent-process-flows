# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
import torch
from torch.utils import data
import numpy as np

TRAIN_SPLIT_PERCENTAGE = 0.7
VAL_SPLIT_PERCENTAGE = 0.8


def get_test_dataset(args, test_batch_size=None):
    """
    Function for getting the dataset for testing

    Parameters:
        args: the arguments from parse_arguments in args
        test_batch_size: batch size used for data

    Returns:
        test_loader: the dataloader for testing
    """
    test_set = BMSequenceSmallBatch(
        data_path=args.data_path,
        batch_size=args.test_batch_size,
        split=args.test_split,
        noise_std=args.noise_std_test,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return test_loader


def get_dataset(args):
    """
    Function for getting the dataset for training and validation

    Parameters:
        args: the arguments from parse_arguments in
        return the dataloader for training and validation

    Returns:
        train_loader: data loader of training data
        val_loader: data loader of validation data
    """
    train_set = BMSequenceBatch(
        data_path=args.data_path, split="train", noise_std=args.noise_std
    )
    val_set = BMSequenceBatch(
        data_path=args.data_path, split="val", noise_std=args.noise_std_test
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, val_loader


class BMSequenceBatch(data.dataset.Dataset):
    """
    Dataset class for observations on irregular time grids of synthetic continuous
    time stochastic processes
    data_path: path to a pickle file storing the data
    split: split of the data, train, val, or test
    """

    def __init__(self, data_path, split="train", noise_std=None):
        super(BMSequenceBatch, self).__init__()
        f = open(data_path, "rb")
        self.data = pickle.load(f)
        f.close()
        self.max_length = 0
        for item in self.data:
            self.max_length = max(len(item), self.max_length)
        total_length = len(self.data)
        train_split = int(total_length * TRAIN_SPLIT_PERCENTAGE)
        val_split = int(total_length * VAL_SPLIT_PERCENTAGE)
        if split == "train":
            self.data = self.data[:train_split]
        elif split == "val":
            self.data = self.data[train_split:val_split]
        elif split == "test":
            self.data = self.data[val_split:]
        self.noise_std = noise_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = np.array(self.data[index])
        padded_values = item[0]
        padded_times = item[1]
        if self.noise_std is not None:
            padded_values = padded_values + self.noise_std * np.random.randn(
                *padded_values.shape
            )
        if len(padded_values.shape) == 2:
            padded_values = np.expand_dims(padded_values, 2)
        return (padded_values.astype(np.float32), padded_times.astype(np.float32))

    @staticmethod
    def preprocess(x):
        padded_values, padded_times = x
        return padded_values[0], padded_times[0], None


class BMSequenceSmallBatch(data.dataset.Dataset):
    """
    Dataset class for observations on irregular time grids of synthetic continuous
    time stochastic processes
    data_path: path to a pickle file storing the data
    split: split of the data, train, val, or test
    """

    def __init__(self, data_path, batch_size=100, split="train", noise_std=None):
        super(BMSequenceSmallBatch, self).__init__()
        f = open(data_path, "rb")
        self.data = pickle.load(f)
        f.close()
        self.max_length = 0
        for item in self.data:
            self.max_length = max(len(item), self.max_length)
        total_length = len(self.data)
        train_split = int(total_length * TRAIN_SPLIT_PERCENTAGE)
        val_split = int(total_length * VAL_SPLIT_PERCENTAGE)
        if split == "train":
            self.data = self.data[:train_split]
        elif split == "val":
            self.data = self.data[train_split:val_split]
        elif split == "test":
            self.data = self.data[val_split:]
        assert 100 % batch_size == 0
        self.batch_size = batch_size
        self.noise_std = noise_std

    def __len__(self):
        return len(self.data) * (100 // self.batch_size)

    def __getitem__(self, index):
        set_index = index // (100 // self.batch_size)
        set_remainder = index % (100 // self.batch_size)
        item = np.array(self.data[set_index])
        padded_values = item[0][
            set_remainder * self.batch_size : (set_remainder + 1) * self.batch_size
        ]
        padded_times = item[1]
        if self.noise_std is not None:
            padded_values = padded_values + self.noise_std * np.random.randn(
                *padded_values.shape
            )
        if len(padded_values.shape) == 2:
            padded_values = np.expand_dims(padded_values, 2)
        return (padded_values.astype(np.float32), padded_times.astype(np.float32))

    @staticmethod
    def preprocess(x):
        padded_values, padded_times = x
        return padded_values[0], padded_times[0], None
