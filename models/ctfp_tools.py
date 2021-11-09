# Copyright (c) 2020-present Royal Bank of Canada
# Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#################################################################
# Code is based on ffjord (https://arxiv.org/abs/1810.01367)
# implementation from https://github.com/rtqichen/ffjordimport
#################################################################

import lib.layers as layers
from .train_misc import set_cnf_options
import os.path as osp
import numpy as np
import torch.nn.functional as F

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams"]


def build_augmented_model_tabular(args, dims, regularization_fns=None):
    """
    The function used for creating conditional Continuous Normlizing Flow
    with augmented neural ODE

    Parameters:
        args: arguments used to create conditional CNF. Check args parser for details.
        dims: dimension of the input. Currently only allow 1-d input.
        regularization_fns: regularizations applied to the ODE function

    Returns:
        a ctfp model based on augmened neural ode
    """
    hidden_dims = tuple(map(int, args.dims.split(",")))
    if args.aug_hidden_dims is not None:
        aug_hidden_dims = tuple(map(int, args.aug_hidden_dims.split(",")))
    else:
        aug_hidden_dims = None

    def build_cnf():
        diffeq = layers.AugODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            effective_shape=args.effective_shape,
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            aug_dim=args.aug_dim,
            aug_mapping=args.aug_mapping,
            aug_hidden_dims=args.aug_hidden_dims,
        )
        odefunc = layers.AugODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
            effective_shape=args.effective_shape,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
            rtol=args.rtol,
            atol=args.atol,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [
            layers.MovingBatchNorm1d(
                dims, bn_lag=args.bn_lag, effective_shape=args.effective_shape
            )
            for _ in range(args.num_blocks)
        ]
        bn_chain = [
            layers.MovingBatchNorm1d(
                dims, bn_lag=args.bn_lag, effective_shape=args.effective_shape
            )
        ]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)
    set_cnf_options(args, model)

    return model
