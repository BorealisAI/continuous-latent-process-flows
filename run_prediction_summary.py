# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pickle

patterns = [
    "samples_mujoco_anode_125_%d.pkl",
    "samples_mujoco_ires_125_%d.pkl",
    "samples_ptb_anode_125_1.pkl",
    "samples_ptb_ires_125_1.pkl",
]

for file_pattern in patterns:

    print(file_pattern)
    if "mujoco" in file_pattern:
        loss_lst = []
        for i in range(1, 11):

            with open(file_pattern % i, "rb") as f:
                pred_lst, target_lst, _ = pickle.load(f)
            loss_lst += [
                np.sqrt(((item1 - item2) ** 2).sum(-1)).flatten()
                for item1, item2 in zip(pred_lst, target_lst)
            ]
        loss_lst = np.concatenate(loss_lst)
    else:
        with open(file_pattern, "rb") as f:
            pred_lst, target_lst, mask_lst = pickle.load(f)
        loss_lst = [
            np.sqrt(((item1 - item2) ** 2).sum(-1)).flatten()[item3.flatten() == 1]
            for item1, item2, item3 in zip(pred_lst, target_lst, mask_lst)
        ]
        loss_lst = [
            np.sqrt(((item1 - item2) ** 2).sum(-1)).flatten()[item3.flatten() == 1]
            for item1, item2, item3 in zip(pred_lst, target_lst, mask_lst)
        ]
        loss_lst = np.concatenate(loss_lst)

    print("L2:", loss_lst.mean())
    print("L2 stdv:", loss_lst.std())
    print("L2 percentile:", np.quantile(loss_lst, [0.25, 0.5, 0.75]))
