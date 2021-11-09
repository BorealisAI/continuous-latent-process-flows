# Copyright (c) 2020-present Royal Bank of Canada
# Copyright (c) 2018 Ricky Tian Qi Chen and Cheng Lu
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

########################################################################
# Code is based on residual flow (https://arxiv.org/abs/1906.02735)
# implementation from https://github.com/rtqichen/residual-flows.
########################################################################

import torch.nn as nn
import torch


class AffineAugSequentialFlow(nn.Module):
    """
    A sequential layer for affine transformation augmented
    normalizing flows
    """

    def __init__(self, layersList, projection_list, effective_dim, aug_model_list=None):
        super(AffineAugSequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)
        if aug_model_list is None:
            self.aug_list = [None] * len(projection_list)
        else:
            self.aug_list = nn.ModuleList(aug_model_list)
        self.projection_list = nn.ModuleList(projection_list)
        self.effective_dim = effective_dim
        assert len(self.chain) % (len(self.aug_list) - 1) == 0
        assert len(self.chain) % (len(self.projection_list) - 1) == 0
        self.interval = len(self.chain) // (len(self.projection_list) - 1)

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self.inverse(x, logpx)
        z = x[:, self.effective_dim :]
        x = x[:, : self.effective_dim]
        logscale_lst = []
        shift_lst = []
        for i, aug_model in enumerate(self.aug_list):
            if aug_model is not None:
                z = aug_model(z)
            scale_shift = self.projection_list[i](z)
            logscale_lst.append(scale_shift[:, : self.effective_dim])
            shift_lst.append(scale_shift[:, self.effective_dim :])

        inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                if i % self.interval == 0:
                    x = (
                        x * torch.exp(logscale_lst[i // self.interval])
                        + shift_lst[i // self.interval]
                    )
                x = self.chain[i](x)
            x = x * torch.exp(logscale_lst[-1]) + shift_lst[-1]
            return x
        else:
            for i in inds:
                if i % self.interval == 0:
                    x = (
                        x * torch.exp(logscale_lst[i // self.interval])
                        + shift_lst[i // self.interval]
                    )
                    logpx = logpx - logscale_lst[i // self.interval].sum(
                        -1, keepdim=True
                    )

                x, logpx = self.chain[i](x, logpx)
            x = x * torch.exp(logscale_lst[-1]) + shift_lst[-1]
            logpx = logpx - logscale_lst[-1].sum(-1, keepdim=True)

            return x, logpx

    def inverse(self, y, logpy=None):
        z = y[:, self.effective_dim :]
        y = y[:, : self.effective_dim]
        logscale_lst = []
        shift_lst = []
        for i, aug_model in enumerate(self.aug_list):
            z = aug_model(z)
            scale_shift = self.projection_list[i](z)
            logscale_lst.append(scale_shift[:, : self.effective_dim])
            shift_lst.append(scale_shift[:, self.effective_dim :])
        if logpy is None:
            y = (y - shift_lst[-1]) * torch.exp(-logscale_lst[-1])
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].inverse(y)
                if i % self.interval == 0:
                    y = (y - shift_lst[i // self.inverval]) * torch.exp(
                        -logscale_lst[i // self.inverval]
                    )
            return y
        else:
            y = (y - shift_lst[-1]) * torch.exp(-logscale_lst[-1])
            logpy = logpy + logscale_lst[-1].sum(-1, keepdim=True)
            for i in range(len(self.chain) - 1, -1, -1):
                y, logpy = self.chain[i].inverse(y, logpy)
                if i % self.interval == 0:
                    y = (y - shift_lst[i // self.interval]) * torch.exp(
                        -logscale_lst[i // self.interval]
                    )
                    logpy = logpy + logscale_lst[i // self.interval].sum(
                        -1, keepdim=True
                    )
            return y, logpy
