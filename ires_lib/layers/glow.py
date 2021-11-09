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
import torch.nn.functional as F


class InvertibleLinear(nn.Module):

    def __init__(self, dim):
        super(InvertibleLinear, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim)[torch.randperm(dim)])

    def forward(self, x, logpx=None):
        y = F.linear(x, self.weight)
        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad

    def inverse(self, y, logpy=None):
        x = F.linear(y, self.weight.inverse())
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.weight)))

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class InvertibleConv2d(nn.Module):

    def __init__(self, dim):
        super(InvertibleConv2d, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim)[torch.randperm(dim)])

    def forward(self, x, logpx=None):
        y = F.conv2d(x, self.weight.view(self.dim, self.dim, 1, 1))
        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad.expand_as(logpx) * x.shape[2] * x.shape[3]

    def inverse(self, y, logpy=None):
        x = F.conv2d(y, self.weight.inverse().view(self.dim, self.dim, 1, 1))
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad.expand_as(logpy) * x.shape[2] * x.shape[3]

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.weight)))

    def extra_repr(self):
        return 'dim={}'.format(self.dim)
