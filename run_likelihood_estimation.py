# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from argparser import parse_arguments
from models.clpf import model_builder
from models.train_misc import get_regularization
import lib.utils as utils
from lib.utils import optimizer_factory, count_parameters, subsample_data
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


def save_model(args, model, optimizer, epoch, itr, path, encoder_optimizer=None):
    torch.save(
        {
            "args": args,
            "state_dict": model.module.state_dict()
            if torch.cuda.is_available() and not args.use_cpu
            else model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "last_epoch": epoch,
            "itr": itr,
            "enc_optim_state_dict": encoder_optimizer.state_dict()
            if encoder_optimizer
            else None,
        },
        path,
    )


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


def train_model(
    data_loader,
    model,
    logger,
    args,
    epoch,
    itr,
    data_preprocess=None,
    tf_writer=None,
    loss_meter=None,
    training_loss_meter=None,
):
    num_exception = 0
    for temp_idx, x in enumerate(data_loader):
        ## x is a tuple of (values, times, stdv, masks)
        optimizer.zero_grad()
        model.zero_grad()
        # cast data and move to device
        masks = None
        lengths = None
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

        if (
            args.data_type in ["real", "unequal"]
            and args.train_subsample_rate is not None
        ):
            values, times, masks = subsample_data(
                args.train_subsample_rate,
                args.train_subsample_random_start,
                values,
                times,
                masks=masks,
                random_subsample_rate=args.train_random_subsample_rate,
            )

        if args.use_gpu:
            values, times = values.cuda(), times.cuda()
            if masks is not None:
                masks = masks.cuda()

        try:
            loss, loss_training, _ = model(times, values, lengths, masks=masks)

            if args.data_type in ["synthetic", "real"]:
                total_length = values.shape[0] * values.shape[1]
            elif args.data_type == "unequal":
                total_length = masks.sum().cpu().data.item()
            else:
                raise NotImplementedError

            loss = loss.sum() / total_length
            loss_training = loss_training.sum() / total_length
            if (
                args.indexed_flow_type == "anode"
                and args.cnf_regularization_coeffs is not None
            ):
                reg_states = get_regularization(model, args.cnf_regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff
                    for reg_state, coeff in zip(
                        reg_states, args.cnf_regularization_coeffs
                    )
                    if coeff != 0
                )
                loss_training = loss_training + reg_loss
            loss_training.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            itr += 1
            if tf_writer is not None:
                tf_writer.add_scalar("train/NLL", loss.cpu().data.item(), itr)
                tf_writer.add_scalar(
                    "train/Loss_Training", loss_training.cpu().data.item(), itr
                )
            if (loss_meter is None) or (training_loss_meter is None):
                if itr % args.log_freq == 0:
                    log_message = "Epoch: {:04d} | Batch: {:04d} | Loss: {:.4f} | Training Loss: {:.4f} ".format(
                        epoch,
                        temp_idx,
                        loss.cpu().data.item(),
                        loss_training.cpu().data.item(),
                    )
                    logger.info(log_message)
            else:
                loss_meter.update(loss.data.cpu().item())
                training_loss_meter.update(loss_training.data.cpu().item())
                if itr % args.log_freq == 0:
                    log_message = "Training Epoch: {:04d} | Batch: {:04d} | Loss: {:.4f} | Loss Avg: {:.4f} | Training Loss: {:.4f} | Training Loss Avg: {:.4f}".format(
                        epoch,
                        temp_idx,
                        loss_meter.val,
                        loss_meter.avg,
                        training_loss_meter.val,
                        training_loss_meter.avg,
                    )
                    logger.info(log_message)
        except:
            num_exception += 1
            print("Run into a exception")
            if num_exception > 0.5 * len(data_loader):
                exit()
            else:
                continue

    return itr


def eval_model(data_loader, model, args, data_preprocess):
    loss_lst = []
    length_lst = []
    max_error = None
    for temp_idx, x in enumerate(data_loader):
        # cast data and move to device
        masks = None
        lengths = None
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

        if (
            args.data_type in ["real", "unequal"]
            and args.eval_subsample_rate is not None
        ):
            values, times, masks = subsample_data(
                args.eval_subsample_rate,
                args.eval_subsample_random_start,
                values,
                times,
                masks=masks,
                random_subsample_rate=args.train_random_subsample_rate,
            )

        if args.use_gpu:
            values, times = values.cuda(), times.cuda()
            if masks is not None:
                masks = masks.cuda()
        loss, _, error = model(times, values, lengths, masks=masks)
        if error is not None:
            error = error.max()

            if max_error is None:
                max_error = error.data.cpu().item()
            else:
                max_error = max(error.data.cpu().item(), max_error)

        loss_lst.append(loss.sum().cpu().data.item())

        if args.data_type in ["synthetic", "real"]:
            length_lst.append(values.shape[0] * values.shape[1])
        elif args.data_type == "unequal":
            length_lst.append(masks.sum().cpu().data.item())
        else:
            raise NotImplementedError

    return sum(np.array(loss_lst)) / sum(length_lst), max_error


if __name__ == "__main__":
    args = parse_arguments()
    if args.np_seed is not None:
        np.random.seed(args.np_seed)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
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
        if optimizer is not None:
            for g in optimizer.param_groups:
                g["lr"] == args.lr

    ## Assign the regularization_coeffs of anode decoder before
    if args.indexed_flow_type == "anode":
        args.cnf_regularization_coeffs = None
        if len(model.decoder.regularization_coeffs) > 0:
            args.cnf_regularization_coeffs = model.decoder.regularization_coeffs
    if args.use_gpu:
        model = torch.nn.DataParallel(model)

    if epoch is not None:
        args.begin_epoch = epoch + 1

    if args.eval:
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
            loss, error = eval_model(test_loader, model, args, data_preprocess)
        print("Evaluation Loss: {:.4f}".format(loss))
        if error is not None:
            print("Max Error: {:.4f}".format(error))
        exit()

    utils.makedirs(args.save)
    utils.makedirs(osp.join(args.save, "save"))
    logger = utils.get_logger(
        logpath=osp.join(args.save, "logs"), filepath=osp.abspath(__file__)
    )

    logger.info(args)
    logger.info("Number of trainable parameters: {}".format(num_params))
    tf_writer = None
    if not args.no_tb_log:
        from tensorboardX import SummaryWriter

        tf_writer = SummaryWriter(osp.join(args.save, "tb_logs"))
        tf_writer.add_text("args", str(args))

    best_loss = float("inf")
    runningave_param = 0.7
    loss_meter = utils.RunningAverageMeter(runningave_param)
    training_loss_meter = utils.RunningAverageMeter(runningave_param)

    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train()
        model.num_iwae = args.num_iwae
        itr = train_model(
            train_loader,
            model,
            logger,
            args,
            epoch,
            itr,
            data_preprocess=data_preprocess,
            tf_writer=tf_writer,
            loss_meter=loss_meter,
            training_loss_meter=training_loss_meter,
        )

        model.eval()
        model.num_iwae = args.niwae_test
        with torch.no_grad():
            loss, error = eval_model(val_loader, model, args, data_preprocess)
        if error is not None:
            log_message = (
                "Validation Epoch : {:04d} | Loss: {:.4f} | Error: {:.4f}".format(
                    epoch, loss, error
                )
            )
        else:
            log_message = "Validation Epoch : {:04d} | Loss: {:.4f}".format(epoch, loss)
        logger.info(log_message)
        if tf_writer is not None:
            tf_writer.add_scalar("valid/NLL", loss, epoch)
            if error is not None:
                tf_writer.add_scalar("valid/Error", error, epoch)
        if loss < best_loss:
            ## Save the best model
            save_model(
                args,
                model,
                optimizer,
                epoch,
                itr,
                osp.join(args.save, "save", "model_best.pth"),
            )
            best_loss = loss

        ## Save the model as current
        save_model(
            args,
            model,
            optimizer,
            epoch,
            itr,
            osp.join(args.save, "save", "model_%d.pth" % epoch),
        )
        ## Save the model as last
        save_model(
            args,
            model,
            optimizer,
            epoch,
            itr,
            osp.join(args.save, "save", "model_last.pth"),
        )
