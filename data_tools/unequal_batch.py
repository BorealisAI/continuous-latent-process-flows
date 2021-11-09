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
import random
import os.path as osp
import math
from scipy import interpolate

## Overriding priority sample_interval > subsample_rate > observ_scale


def get_unequal_dataset(args):
    train_set = UnEqualBatchSequence(
        args.data_path,
        args.batch_size,
        args.max_length,
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
    val_set = UnEqualBatchSequence(
        args.data_path,
        args.test_batch_size,
        args.max_length,
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


def get_unequal_test_dataset(args):
    test_set = UnEqualBatchSequence(
        args.data_path,
        args.test_batch_size,
        args.max_length,
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


class UnEqualBatchSequence(data.dataset.Dataset):
    """
    Given data with fixed length. Generate mini batches of data with irregular time scale
    """

    def __init__(
        self,
        data_path,
        batch_size,
        max_length,
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
        super(UnEqualBatchSequence, self).__init__()
        if split == "test":
            data_path = osp.join(data_path, "test_p3.pkl")
        elif split == "val":
            data_path = osp.join(data_path, "valid_p3.pkl")
        else:
            data_path = osp.join(data_path, "train_p3.pkl")

        f = open(data_path, "rb")
        self.data = pickle.load(f)
        f.close()
        self.max_length = 0
        self.sample_interval = sample_interval
        self.subsample_rate = subsample_rate
        self.batch_size = batch_size
        self.maximum_time = maximum_time
        self.bias = bias
        self.shuffle = shuffle
        self.scale = scale
        self.max_length = max_length
        self.interpolation = interpolation
        self.noise_std = noise_std
        if (self.sample_interval is None) and (self.subsample_rate is not None):
            self.sample_interval = (
                float(self.maximum_time) / self.max_length * self.subsample_rate
            )

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def pad_to_same_length(self, sequences):
        data_dim = sequences[0].shape[-1]
        batch_max_length = max([item.shape[0] for item in sequences])
        padded_sequences = []
        padded_masks = []
        for item in sequences:
            seq_length = item.shape[0]
            padded_item = np.concatenate(
                [
                    item,
                    np.zeros(
                        (batch_max_length - seq_length, data_dim), dtype=np.float32
                    )
                    + item[-1],
                ],
                axis=0,
            )
            padded_sequences.append(padded_item)
            padded_masks.append(
                np.array(
                    [1] * seq_length + [0] * (batch_max_length - seq_length)
                ).astype(np.float32)
            )
        padded_sequences = np.stack(padded_sequences)
        padded_masks = np.stack(padded_masks)
        batch_max_length = min(batch_max_length, self.max_length)
        return (
            padded_sequences[:, :batch_max_length],
            padded_masks[:, :batch_max_length],
            batch_max_length,
        )

    def __sample_time_idx__(self, item_len):
        ## Assume that the observation times are sampled from process approximated by poisson
        ## with fixed intensity
        prev_time = 0
        if self.sample_interval is not None:
            prev_time = (
                float(self.maximum_time) / (2 * self.max_length) - self.sample_interval
            )
        times = []
        indices = []
        while True:
            if self.sample_interval is None:
                time_interval = np.random.exponential(self.scale)
            else:
                time_interval = self.sample_interval
            time = prev_time + time_interval
            if time > self.maximum_time:
                break

            if self.interpolation:
                index = time / float(self.maximum_time) * self.max_length
            else:
                index = int(time / float(self.maximum_time) * self.max_length)

            if (index >= item_len) or (self.interpolation and index + 1 > item_len):
                break
            times.append(time + self.bias)
            indices.append(index)
            prev_time = time
        return np.array(times), indices

    def __getitem__(self, index):
        if self.shuffle:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            indices = indices[: self.batch_size]
            items = [self.data[index] for index in indices]
        else:
            begin_idx = index * self.batch_size
            end_idx = begin_idx + self.batch_size
            items = self.data[begin_idx:end_idx]
        # Items of shape batchsize * length * data_dime
        item_values, item_masks, batch_max_length = self.pad_to_same_length(items)
        assert batch_max_length <= self.max_length
        item_len = item_values.shape[1]
        item_times, indices = self.__sample_time_idx__(item_len)
        if self.interpolation:
            interpolated_items = interpolate.interp1d(
                np.arange(0, item_len), item_values, axis=1
            )
            item_values = interpolated_items(np.array(indices))
            interpolated_masks = interpolate.interp1d(
                np.arange(0, item_len), item_masks, axis=1
            )
            item_masks = interpolated_masks(np.array(indices))
            item_masks[item_masks < 1] = 0
            item_masks_repeated = np.expand_dims(item_masks, -1)
            item_masks_repeated = np.repeat(
                item_masks_repeated, item_values.shape[-1], axis=-1
            )
            item_values[item_masks == 0] = 0

        else:
            item_values = item_values[:, indices]
            item_masks = item_masks[:, indices]
        # sample_time and get idx

        if self.noise_std is not None:
            item_values = item_values + self.noise_std * np.random.randn(
                *item_values.shape
            )

        padded_times = torch.Tensor(item_times).type(torch.FloatTensor)
        padded_masks = torch.Tensor(item_masks).type(torch.FloatTensor)
        padded_values = torch.Tensor(item_values).type(torch.FloatTensor)

        return (padded_values, padded_times, padded_masks)

    @staticmethod
    def preprocess(x):
        padded_values, padded_times, padded_masks = x
        padded_values, padded_times, padded_masks = (
            padded_values[0],
            padded_times[0],
            padded_masks[0],
        )

        return padded_values, padded_times, padded_masks
