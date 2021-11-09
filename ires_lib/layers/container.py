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

import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None):
        if logpx is None:
            for i in range(len(self.chain)):
                x = self.chain[i](x)
            return x
        else:
            for i in range(len(self.chain)):
                x, logpx = self.chain[i](x, logpx)
            return x, logpx

    def inverse(self, y, logpy=None):
        if logpy is None:
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].inverse(y)
            return y
        else:
            for i in range(len(self.chain) - 1, -1, -1):
                y, logpy = self.chain[i].inverse(y, logpy)
            return y, logpy


class Inverse(nn.Module):

    def __init__(self, flow):
        super(Inverse, self).__init__()
        self.flow = flow

    def forward(self, x, logpx=None):
        return self.flow.inverse(x, logpx)

    def inverse(self, y, logpy=None):
        return self.flow.forward(y, logpy)
