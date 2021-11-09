# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pickle

time_limit = 30
num_batch = 100
batch_size = 100
samples = []
initial_value = 0
scale = 0.5
step_size = 1e-5

## drift and variance gaussian markov
sigma_value = 0.2
a = 0.5
b = 0.2


def drift(value, t):
    global a, b
    return a * np.sin(t) * value + b * np.cos(t)


def sigma(value, t):
    return sigma_value / (1 + np.exp(-t))


def simulate_one_batch(time_stamps):
    batch_values = []
    global initial_value
    global step_size
    global batch_size
    value = initial_value * np.ones(batch_size)
    time = 0
    time_stamp_index = 0
    global time_limit
    while True:
        if time + step_size > time_stamps[time_stamp_index]:
            time_diff = time_stamps[time_stamp_index] - time
            value_diff = drift(value, time) * time_diff + sigma(
                value, time
            ) * np.random.randn(batch_size) * np.sqrt(time_diff)
            value = value + value_diff
            batch_values.append(value)
            time = time_stamps[time_stamp_index]
            time_stamp_index += 1
            if time_stamp_index == len(time_stamps):
                return batch_values
        else:
            value_diff = drift(value, time) * step_size + sigma(
                value, time
            ) * np.random.randn(batch_size) * np.sqrt(step_size)
            value = value + value_diff
            time = time + step_size


if __name__ == "__main__":
    for _ in range(num_batch):
        time = 0
        sequences = []
        time_intervals = []
        time_stamps = []
        while True:
            time_interval = np.random.exponential(scale)
            time = time + time_interval
            if time > time_limit:
                break
            time_intervals.append(time_interval)
            time_stamps.append(time)

        sequences = np.array(simulate_one_batch(time_stamps))
        sequences = np.transpose(sequences)
        samples.append((np.array(sequences), np.array(time_stamps)))

    with open("lsde_05.pkl", "wb") as f:
        pickle.dump(samples, f)
