# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pickle

time_limit = 2
num_batch = 100
batch_size = 100
samples = []
initial_value = 1
scale = 0.05
step_size = 5e-7

## drift and variance gaussian markov
sigma_value = 10
rho = 28
beta = 8 / 3

alpha_x = 0.1
alpha_y = 0.28
alpha_z = 0.3
alpha = np.array([alpha_x, alpha_y, alpha_z])


def drift(value, t):
    x = value[:, 0:1]
    y = value[:, 1:2]
    z = value[:, 2:3]
    x_drift = sigma_value * (y - x)
    y_drift = x * (rho - z) - y
    z_drift = x * y - beta * z
    return np.concatenate([x_drift, y_drift, z_drift], axis=1)


def sigma(value, t):
    return alpha


def simulate_one_batch(time_stamps):
    batch_values = []
    global initial_value
    global step_size
    global batch_size
    value = initial_value * np.random.randn(batch_size, 3)
    time = 0
    time_stamp_index = 0
    global time_limit
    while True:
        if time + step_size > time_stamps[time_stamp_index]:
            time_diff = time_stamps[time_stamp_index] - time
            value_diff = drift(value, time) * time_diff + sigma(
                value, time
            ) * np.random.randn() * np.sqrt(time_diff)
            value = value + value_diff
            batch_values.append(value)
            time = time_stamps[time_stamp_index]
            time_stamp_index += 1
            if time_stamp_index == len(time_stamps):
                return batch_values
        else:
            value_diff = drift(value, time) * step_size + sigma(
                value, time
            ) * np.random.randn() * np.sqrt(step_size)
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
        sequences = np.transpose(sequences, (1, 0, 2))
        samples.append((np.array(sequences), np.array(time_stamps)))

    with open("lorenz_curve_005.pkl", "wb") as f:
        pickle.dump(samples, f)
