# Copyright (c) 2019 Ricky Tian Qi Chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################
# Residual Flows for Invertible Generative Modeling 
# Author: Ricky T. Q. Chen
# Link: https://github.com/rtqichen/residual-flows
###################################################

import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ['MovingBatchNorm1d', 'MovingBatchNorm2d']


class MovingBatchNormNd(nn.Module):

    def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0., affine=True):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        if self.affine:
            self.bias.data.zero_()

    def forward(self, x, logpx=None):
        c = x.size(1)
        used_mean = self.running_mean.clone().detach()

        if self.training:
            # compute batch statistics
            x_t = x.transpose(0, 1).contiguous().view(c, -1)
            batch_mean = torch.mean(x_t, dim=1)

            # moving average
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= (1. - self.bn_lag**(self.step[0] + 1))

            # update running estimates
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.step += 1

        # perform normalization
        used_mean = used_mean.view(*self.shape).expand_as(x)

        y = x - used_mean

        if self.affine:
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y + bias

        if logpx is None:
            return y
        else:
            return y, logpx

    def inverse(self, y, logpy=None):
        used_mean = self.running_mean

        if self.affine:
            bias = self.bias.view(*self.shape).expand_as(y)
            y = y - bias

        used_mean = used_mean.view(*self.shape).expand_as(y)
        x = y + used_mean

        if logpy is None:
            return x
        else:
            return x, logpy

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},'
            ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)
        )


class MovingBatchNorm1d(MovingBatchNormNd):

    @property
    def shape(self):
        return [1, -1]


class MovingBatchNorm2d(MovingBatchNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]
