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
initial_value = 1
scale = 0.05

## Brownian motion with drift and variance
sigma = 0.2
mu = 0.1

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

        for _ in range(batch_size):
            time = 0
            value = initial_value
            sequence = []
            for time_interval in time_intervals:
                z = np.random.randn()
                value = value * np.exp(
                    (mu - sigma * sigma * 0.5) * time_interval
                    + sigma * np.sqrt(time_interval) * z
                )
                sequence.append(value)
            sequences.append(np.array(sequence))

        samples.append((np.array(sequences), np.array(time_stamps)))

    with open("gbm_005.pkl", "wb") as f:
        pickle.dump(samples, f)
