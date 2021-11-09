# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pickle

time_limit = 15
num_batch = 100
batch_size = 100
samples = []
initial_value = 0
scale = 0.05
step_size = 1e-5

## drift and variance gaussian markov
b = 0.002
a_1, a_2, a_3, a_4 = 0.002, 0.005, -0.003, -0.002
a_line = np.array([[a_1, a_2, a_3, a_4]])
identity = np.concatenate([np.zeros([3, 1]), np.identity(3)], 1)
a = np.concatenate([identity, a_line], 0)
a_T = np.transpose(a)
sigma_value = np.array([0, 0, 0, 0.001])


def drift(value, t):
    global a_T
    return np.matmul(value, a_T)


def sigma(value, t):
    global sigma_value
    return sigma_value


def simulate_one_batch(time_stamps):
    batch_values = []
    global initial_value
    global step_size
    global batch_size
    value = np.random.randn(batch_size, 4)
    time = 0
    time_stamp_index = 0
    global time_limit
    while True:
        if time + step_size > time_stamps[time_stamp_index]:
            time_diff = time_stamps[time_stamp_index] - time
            value_diff = drift(value, time) * time_diff + sigma(
                value, time
            ) * np.random.randn(batch_size, 4) * np.sqrt(time_diff)
            value = value + value_diff
            batch_values.append(value)
            time = time_stamps[time_stamp_index]
            time_stamp_index += 1
            if time_stamp_index == len(time_stamps):
                return batch_values
        else:
            value_diff = drift(value, time) * step_size + sigma(
                value, time
            ) * np.random.randn(batch_size, 4) * np.sqrt(step_size)
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

        sequences = np.array(simulate_one_batch(time_stamps))[:, :, 0]
        sequences = np.transpose(sequences)
        samples.append((np.array(sequences), np.array(time_stamps)))

    with open("car_005.pkl", "wb") as f:
        pickle.dump(samples, f)
