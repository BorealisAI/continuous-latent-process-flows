# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from argparser import parse_arguments
from models.clpf import model_builder
from lib.utils import optimizer_factory, count_parameters
from data_tools.bm_sequential_batch import (
    BMSequenceBatch,
    get_dataset,
    get_test_dataset,
)
from data_tools.batch_sequence import get_real_dataset, get_real_test_dataset
from data_tools.unequal_batch import (
    get_unequal_dataset,
    get_unequal_test_dataset,
    UnEqualBatchSequence,
)

import torch
import os.path as osp
import numpy as np
import pickle

NUM_SAMPLES = 10


def preprocess_data_forpred(values, times, masks=None, mode="pred"):
    lengths = values.shape[1]
    # Mask is either none or a binary tensor of shape batch_size x max_step
    if mode == "pred":
        if lengths <= 2:
            return None, None
        ts = [(times[0], times[0:1], 0), (times[1], times[1:2], 0)]
        target_value_lst = []
        for i in range(2, lengths - 1):
            c_time_tuple = (times[i], times[i : i + 2], 0)
            ts.append(c_time_tuple)
            target_value_lst.append(values[:, i + 1])
        if masks is not None:
            masks = [masks[:, i] for i in range(3, lengths)]
        return ts, target_value_lst, values, masks
    elif mode == "interp":
        processed_value_list = []
        observed_indices = list(range(0, lengths, 2))
        observed_length = len(observed_indices)
        if observed_length < 2:
            return None, None
        ts = [
            (
                times[observed_indices[0]],
                times[observed_indices[0] : observed_indices[0] + 1],
                0,
            ),
            (
                times[observed_indices[1]],
                times[observed_indices[1] : observed_indices[1] + 1],
                0,
            ),
        ]
        target_value_lst = []
        for i in range(2, observed_length):
            c_time_tuple = (
                times[observed_indices[i]],
                times[observed_indices[i] - 1 : observed_indices[i] + 1],
                1,
            )
            target_value_lst.append(values[:, observed_indices[i] - 1])
            ts.append(c_time_tuple)
        values = values[:, observed_indices]
        if masks is not None:
            masks = [
                masks[:, observed_indices[i] - 1] * masks[:, observed_indices[i]]
                for i in range(2, observed_length)
            ]
        return ts, target_value_lst, values, masks


def load_model(resume, model, optimizer=None):
    iter = 0
    epoch = None
    checkpt = torch.load(resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])

    if (optimizer is not None) and ("optim_state_dict" in checkpt.keys()):
        optimizer.load_state_dict(checkpt["optim_state_dict"])

    if "last_epoch" in checkpt.keys():
        epoch = checkpt["last_epoch"]

    if "iter" in checkpt.keys():
        iter = checkpoint["iter"]

    return epoch, iter


def check_observation_grids(time, max_time, time_gap):
    time = time.data.cpu().numpy()
    if max_time - time[-1] <= time_gap:
        return False
    time_prev = np.zeros_like(time)
    time_prev[1:] = time[:-1]
    time_gaps = time - time_prev
    min_gap = np.min(time_gaps)
    if min_gap < time_gap:
        return False
    return True


def prediction(data_loader, model, args, data_preprocess):
    loss_lst = []
    length_lst = []
    pred_lst = []
    target_lst = []
    mask_lst = []

    for temp_idx, x in enumerate(data_loader):
        # cast data and move to device
        masks = None
        if data_preprocess is None:
            values, times, lengths = x
        else:
            if args.data_type in ["synthetic", "real"]:
                values, times, lengths = data_preprocess(x)
                times = torch.stack([times for i in range(values.shape[0])])
            elif args.data_type == "unequal":
                values, times, masks = data_preprocess(x)
                times = torch.stack([times for i in range(values.shape[0])])
            else:
                raise NotImplementedError

        if args.use_gpu:
            values, times = values.cuda(), times.cuda()
            if masks is not None:
                masks = masks.cuda()
        # compute loss
        times = times[0]
        ts, target_values, values, masks = preprocess_data_forpred(
            values, times, masks, args.pred_mode
        )

        if ts is not None:
            samples = model.sample(
                None,
                batch_size=args.niwae_test,
                ts=ts,
                inputs=values,
                observe_sample_size=args.observe_sample_size,
            )
            ## Compute the loss
            samples_cat = torch.stack(samples)
            # number of targets x batch size x number of samples x data dimension
            pred = samples_cat.mean(2)
            # tensor of size number of targets x batch size x data dimension
            targets_cat = torch.stack(target_values)
            # tensor of size number of targets x batch size x data dimension
            if masks is None:
                loss = (
                    torch.sqrt(((pred - targets_cat) ** 2).sum(-1)).cpu().data.numpy()
                )
                loss_lst.append(loss)
                length_lst.append(pred.shape[0] * pred.shape[1])
            else:
                masks_stacked = torch.stack(masks)
                loss = (
                    (torch.sqrt(((pred - targets_cat) ** 2).sum(-1)) * masks_stacked)
                    .cpu()
                    .data.numpy()
                )
                loss_lst.append(loss)
                length_lst.append(
                    sum([item.sum().data.cpu().numpy() for item in masks])
                )
                mask_lst.append(masks_stacked.data.cpu().numpy())
            pred_lst.append(pred.data.cpu().numpy())
            target_lst.append(targets_cat.data.cpu().numpy())

    return loss_lst, length_lst, pred_lst, target_lst, mask_lst


if __name__ == "__main__":
    args = parse_arguments()
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    if args.np_seed is not None:
        np.random.seed(args.np_seed)

    model = model_builder(args)
    use_gpu = torch.cuda.is_available() and (not args.use_cpu)
    args.use_gpu = use_gpu
    if args.use_gpu:
        model = model.cuda()
    ## Put the data preprocessing tools
    if args.data_type == "synthetic":
        train_loader, val_loader = get_dataset(args)
    elif args.data_type == "real":
        train_loader, val_loader = get_real_dataset(args)
    elif args.data_type == "unequal":
        train_loader, val_loader = get_unequal_dataset(args)
    else:
        raise NotImplementedError

    if args.data_type in ["synthetic", "real"]:
        data_preprocess = BMSequenceBatch.preprocess
    elif args.data_type == "unequal":
        data_preprocess = UnEqualBatchSequence.preprocess
    else:
        data_preprocess = None

    ## Define parameters and optimizer
    optimizer = None
    if not args.eval:
        trainable_parameters = model.parameters()
        optimizer, num_params = optimizer_factory(args, trainable_parameters)
    else:
        num_params = count_parameters(model)

    ## reload model
    itr = 0
    epoch = None
    if args.resume is not None:
        epoch, itr = load_model(args.resume, model, optimizer=optimizer)
        print("Reload model from epoch:", epoch)

    ## Assign the regularization_coeffs of anode decoder before
    if args.indexed_flow_type == "anode":
        args.cnf_regularization_coeffs = None
        if len(model.decoder.regularization_coeffs) > 0:
            args.cnf_regularization_coeffs = model.decoder.regularization_coeffs

    if epoch is not None:
        args.begin_epoch = epoch + 1

    print("Number of trainable parameters: {}".format(num_params))
    model.eval()
    model.num_iwae = args.niwae_test
    if args.data_type == "synthetic":
        test_loader = get_test_dataset(args)
    elif args.data_type == "real":
        test_loader = get_real_test_dataset(args)
    elif args.data_type == "unequal":
        test_loader = get_unequal_test_dataset(args)
    else:
        raise NotImplementedError
    with torch.no_grad():
        loss_lst, length_lst, pred_lst, target_lst, mask_lst = prediction(
            test_loader, model, args, data_preprocess
        )
        print("Loss sum:", sum([item.sum() for item in loss_lst]))
        print("Length:", sum(length_lst))
        ## Print the standard deviation
        if args.data_type == "unequal":
            loss_list = [
                item1.flatten()[item2.flatten() == 1]
                for item1, item2 in zip(loss_lst, mask_lst)
            ]
        else:
            loss_list = [item1.flatten() for item1 in loss_lst]

        loss_list = np.concatenate(loss_lst)
        print("RMSE:", sum([item.sum() for item in loss_lst]) / sum(length_lst))
        print("RMSE2: ", np.mean(loss_list))
        print("RMSE Stdv: ", np.std(loss_list))
        if args.save_np is not None:
            with open(args.save_np, "wb") as f:
                pickle.dump((pred_lst, target_lst, mask_lst), f)
