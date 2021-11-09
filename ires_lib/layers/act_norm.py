# MIT License

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

__all__ = ['ActNorm1d', 'ActNorm2d']


class ActNormNd(nn.Module):

    def __init__(self, num_features, eps=1e-12):
        super(ActNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('initialized', torch.tensor(0))

    @property
    def shape(self):
        raise NotImplementedError

    def forward(self, x, logpx=None):
        c = x.size(1)

        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, 1).contiguous().view(c, -1)
                batch_mean = torch.mean(x_t, dim=1)
                batch_var = torch.var(x_t, dim=1)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.view(*self.shape).expand_as(x)
        weight = self.weight.view(*self.shape).expand_as(x)

        y = (x + bias) * torch.exp(weight)

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x)

    def inverse(self, y, logpy=None):
        assert self.initialized
        bias = self.bias.view(*self.shape).expand_as(y)
        weight = self.weight.view(*self.shape).expand_as(y)

        x = y * torch.exp(-weight) - bias

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)

    def _logdetgrad(self, x):
        return self.weight.view(*self.shape).expand(*x.size()).contiguous().view(x.size(0), -1).sum(1, keepdim=True)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))


class ActNorm1d(ActNormNd):

    @property
    def shape(self):
        return [1, -1]


class ActNorm2d(ActNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]
