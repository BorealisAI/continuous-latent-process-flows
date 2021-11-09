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

__all__ = ['SqueezeLayer']


class SqueezeLayer(nn.Module):

    def __init__(self, downscale_factor):
        super(SqueezeLayer, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x, logpx=None):
        squeeze_x = squeeze(x, self.downscale_factor)
        if logpx is None:
            return squeeze_x
        else:
            return squeeze_x, logpx

    def inverse(self, y, logpy=None):
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        if logpy is None:
            return unsqueeze_y
        else:
            return unsqueeze_y, logpy


def unsqueeze(input, upscale_factor=2):
    return torch.pixel_shuffle(input, upscale_factor)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.reshape(batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor)

    output = input_view.permute(0, 1, 3, 5, 2, 4)
    return output.reshape(batch_size, out_channels, out_height, out_width)
