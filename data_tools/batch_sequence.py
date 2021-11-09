## Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
import torch
from torch.utils import data
import numpy as np
import random
import math
from scipy import interpolate


def get_real_dataset(args):
    train_set = BatchSequence(
        args.data_path,
        args.batch_size,
        args.max_time,
        args.time_bias,
        args.observ_scale,
        shuffle=True,
        split="train",
        sample_interval=args.observ_interval,
        interpolation=args.interp_observ,
        noise_std=args.noise_std,
        subsample_rate=args.observe_subsample_rate,
    )

    val_set = BatchSequence(
        args.data_path,
        args.test_batch_size,
        args.max_time,
        args.time_bias,
        args.observ_scale,
        shuffle=False,
        split="val",
        sample_interval=args.observ_interval,
        interpolation=args.interp_observ,
        noise_std=args.noise_std_test,
        subsample_rate=args.observe_subsample_rate,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=1, shuffle=False, drop_last=True
    )
    return train_loader, val_loader


def get_real_test_dataset(args):
    test_set = BatchSequence(
        args.data_path,
        args.test_batch_size,
        args.max_time,
        args.time_bias,
        args.observ_scale,
        shuffle=False,
        split=args.test_split,
        sample_interval=args.observ_interval,
        interpolation=args.interp_observ,
        noise_std=args.noise_std_test,
        subsample_rate=args.observe_subsample_rate,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, drop_last=True
    )
    return test_loader


class BatchSequence(data.dataset.Dataset):
    """
    Given data with fixed length. Generate mini batches of data with irregular time scale
    """

    def __init__(
        self,
        data_path,
        batch_size,
        maximum_time=30,
        bias=0.2,
        scale=2,
        shuffle=False,
        split="train",
        sample_interval=None,
        interpolation=False,
        noise_std=None,
        subsample_rate=None,
    ):
        super(BatchSequence, self).__init__()
        f = open(data_path, "rb")
        self.data = pickle.load(f)
        f.close()
        self.max_length = 0
        self.sample_interval = sample_interval
        self.subsample_rate = subsample_rate

        trainsplit = int(len(self.data) * 0.7)
        valsplit = int(len(self.data) * 0.8)
        self.batch_size = batch_size
        self.maximum_time = maximum_time
        self.bias = bias
        self.shuffle = shuffle
        self.scale = scale
        self.interpolation = interpolation
        if split == "train":
            self.data = self.data[:trainsplit]
        elif split == "val":
            self.data = self.data[trainsplit:valsplit]
        elif split == "test":
            self.data = self.data[valsplit:]
        self.noise_std = noise_std
        if (self.sample_interval is None) and (self.subsample_rate is not None):
            self.sample_interval = (
                float(self.maximum_time) / self.data.shape[1] * self.subsample_rate
            )

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def __sample_time_idx__(self, item_len):
        ## Assume that the observation times are sampled from process approximated by poisson
        ## with fixed intensity
        prev_time = 0
        if self.sample_interval is not None:
            prev_time = float(self.maximum_time) / (2 * item_len) - self.sample_interval
        times = []
        indices = []
        while True:
            if self.sample_interval is None:
                time_interval = np.random.exponential(self.scale)
            else:
                time_interval = self.sample_interval
            time = prev_time + time_interval
            if np.float32(time + self.bias) == np.float32(prev_time + self.bias):
                time = time + 1e-5
            if time > self.maximum_time:
                break

            if self.interpolation:
                index = time / float(self.maximum_time) * item_len
            else:
                index = int(time / float(self.maximum_time) * item_len)

            if (index >= item_len) or (self.interpolation and index + 1 > item_len):
                break
            times.append(time + self.bias)
            indices.append(index)
            prev_time = time
        return np.array(times), indices

    def __getitem__(self, index):
        if self.shuffle:
            begin_idx = random.randint(0, len(self.data) - self.batch_size)
            end_idx = begin_idx + self.batch_size
        else:
            begin_idx = index * self.batch_size
            end_idx = begin_idx + self.batch_size
        items = self.data[begin_idx:end_idx]
        # Items of shape batchsize * length * data_dime
        item_len = items.shape[1]
        item_times, indices = self.__sample_time_idx__(item_len)
        if self.interpolation:
            interpolated_items = interpolate.interp1d(
                np.arange(0, item_len), items, axis=1
            )
            item_values = interpolated_items(np.array(indices))
        else:
            item_values = items[:, indices]

        if self.noise_std is not None:
            item_values = item_values + self.noise_std * np.random.randn(
                *item_values.shape
            )
        padded_times = torch.Tensor(item_times).type(torch.FloatTensor)
        padded_values = torch.Tensor(item_values).type(torch.FloatTensor)
        return (padded_values, padded_times)
