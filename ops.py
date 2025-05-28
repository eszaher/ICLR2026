import math
import random
from collections import abc
import os
import torch
from torch import autograd
from torch import nn
from torch.nn import functional as F
from torch.nn import Embedding as Embedding
from torch.autograd import Function
from torch.utils.cpp_extension import load
import contextlib
import warnings

# Helper OPs


import contextlib
import warnings

import torch
from torch import autograd
from torch.nn import functional as F

enabled = True
#enabled = False

weight_gradients_disabled = False
#weight_gradients_disabled = True


@contextlib.contextmanager
def no_weight_gradients():
    global weight_gradients_disabled

    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if could_use_op(input):
        return conv2d_gradfix(
            transpose=False,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=0,
            dilation=dilation,
            groups=groups,
        ).apply(input, weight, bias)

    return F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    if could_use_op(input):
        return conv2d_gradfix(
            transpose=True,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        ).apply(input, weight, bias)

    return F.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def could_use_op(input):
    if (not enabled) or (not torch.backends.cudnn.enabled):
        return False

    if input.device.type != "cuda":
        return False
    
    
    try:
        parts = torch.__version__.split('.')
        major = int(parts[0])
        minor = int(parts[1])
        if major >= 1 or (major == 1 and minor >= 7):
            return True
    except (ValueError):
        pass
    warnings.warn(
        f"conv2d_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.conv2d()."
    )

    return False


def ensure_tuple(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim

    return xs


conv2d_gradfix_cache = dict()


def conv2d_gradfix(
    transpose, weight_shape, stride, padding, output_padding, dilation, groups
):
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = ensure_tuple(stride, ndim)
    padding = ensure_tuple(padding, ndim)
    output_padding = ensure_tuple(output_padding, ndim)
    dilation = ensure_tuple(dilation, ndim)

    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in conv2d_gradfix_cache:
        return conv2d_gradfix_cache[key]

    common_kwargs = dict(
        stride=stride, padding=padding, dilation=dilation, groups=groups
    )

    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]

        return [
            input_shape[i + 2]
            - (output_shape[i + 2] - 1) * stride[i]
            - (1 - 2 * padding[i])
            - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    class Conv2d(autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            if not transpose:
                out = F.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)

            else:
                out = F.conv_transpose2d(
                    input=input,
                    weight=weight,
                    bias=bias,
                    output_padding=output_padding,
                    **common_kwargs,
                )

            ctx.save_for_backward(input, weight, bias)

            return out

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            grad_input, grad_weight, grad_bias = None, None, None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(
                    input_shape=input.shape, output_shape=grad_output.shape
                )
                grad_input = conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs,
                ).apply(grad_output, weight, None)

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input, weight, bias)

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3))

            return grad_input, grad_weight, grad_bias

    class Conv2dGradWeight(autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, input, weight, bias):
            """op = torch._C._jit_get_operation(
                "aten::cudnn_convolution_backward_weight"
                if not transpose
                else "aten::cudnn_convolution_transpose_backward_weight"
            )
            flags = [
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32,
            ]"""
            """grad_weight = op(
                weight_shape,
                grad_output,
                input,
                padding,
                stride,
                dilation,
                groups,
                *flags,
            )"""
            #_, weight = ctx.saved_tensors #NEW
            grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
                grad_output,
                input,
                weight,
                None if bias is None else list(bias.shape),  # Use bias shape or None
                list(stride),  # Ensure stride is a list
                list(padding),  # Ensure padding is a list
                list(dilation),  # Ensure dilation is a list
                transpose,  # Whether the operation is transposed
                [0, 0] if not transpose else list(output_padding),  # Correct output_padding
                groups,  # Number of groups
                [False, True, bias is not None],  # Output mask for gradients
            )
            ctx.save_for_backward(grad_output, input)

            return grad_weight

        @staticmethod
        def backward(ctx, grad_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad_grad_output, grad_grad_input = None, None
            grad_weight = None # remove this

            if ctx.needs_input_grad[0]:
                grad_grad_output = Conv2d.apply(input, grad_grad_weight, None)

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(
                    input_shape=input.shape, output_shape=grad_output.shape
                )
                grad_grad_input = conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs,
                ).apply(grad_output, grad_grad_weight, None)

            return grad_grad_output, grad_grad_input, None, None # remove last 2

    conv2d_gradfix_cache[key] = Conv2d

    return Conv2d
    


def upfirdn2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    up: int or tuple = 1,
    down: int or tuple = 1,
    pad: tuple = (0, 0)
) -> torch.Tensor:
    """
    A pure-Python/PyTorch functional implementation of 'upfirdn2d':
      1) Upsample by factor (up_y, up_x).
      2) Convolve with a 2D FIR filter (kernel).
      3) Downsample by factor (down_y, down_x).
      4) Pad or crop (as specified by pad_x0, pad_x1, pad_y0, pad_y1).

    Args:
        x (Tensor):
            Input tensor of shape (B, C, H, W).
        kernel (Tensor):
            2D FIR filter kernel, shape (kh, kw).
        up (int or (int, int)):
            Upsampling factor (if int, both dims are the same). 
            Default is 1 (no upsampling).
        down (int or (int, int)):
            Downsampling factor (if int, both dims are the same). 
            Default is 1 (no downsampling).
        pad (tuple):
            Padding tuple. If length is 2, it's treated as symmetric 
            (pad_x0=pad_y0=pad[0], pad_x1=pad_y1=pad[1]).
            If length is 4, the order is (pad_x0, pad_x1, pad_y0, pad_y1).

    Returns:
        Tensor of shape (B, C, outH, outW).
    """
    # Ensure up/down are tuples
    if not isinstance(up, (tuple, list)):
        up = (up, up)
    if not isinstance(down, (tuple, list)):
        down = (down, down)

    # If pad has length 2, treat them as symmetrical for x and y
    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    up_x, up_y = up
    down_x, down_y = down
    pad_x0, pad_x1, pad_y0, pad_y1 = pad

    # Cache shapes
    b, c, in_h, in_w = x.shape
    kernel_h, kernel_w = kernel.shape

    # Reshape to [BC, H, W, 1] so we can do "inserting zeros" easily
    x = x.reshape(-1, in_h, in_w, 1)  # [B*C, H, W, 1]

    # 1) Upsample by inserting zeros
    x = x.view(-1, in_h, 1, in_w, 1, 1)  # [BC, H, 1, W, 1, 1]
    x = F.pad(x, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])  # Insert zeros
    x = x.view(-1, in_h * up_y, in_w * up_x, 1)  # [BC, upH, upW, 1]

    # 2) Pad for boundary conditions
    x = F.pad(
        x,
        [0, 0, max(pad_x0, 0), max(pad_x1, 0),
         max(pad_y0, 0), max(pad_y1, 0)]
    )

    # Crop if negative padding
    x = x[
        :,
        max(-pad_y0, 0) : x.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : x.shape[2] - max(-pad_x1, 0),
        :
    ]

    # 3) Convolve with flipped kernel
    x = x.permute(0, 3, 1, 2)  # to NCHW => [BC, 1, H, W]
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    x = F.conv2d(x, w, stride=1, padding=0, groups=1)  # depthwise

    # Now shape is [BC, 1, outH, outW], rearr -> [BC, outH, outW, 1]
    x = x.permute(0, 2, 3, 1)

    # 4) Downsample
    x = x[:, ::down_y, ::down_x, :]

    # Compute final out shape
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    # Reshape back to (B, C, outH, outW)
    x = x.view(b, c, out_h, out_w)

    return x


def fused_leaky_relu(
    input: torch.Tensor,
    bias: torch.Tensor = None,
    negative_slope: float = 0.2,
    scale: float = 2 ** 0.5,
) -> torch.Tensor:
    """
    Pure-Python functional version of fused LeakyReLU:
      1) Optionally add bias (broadcast across batch & spatial dims).
      2) Apply LeakyReLU with `negative_slope`.
      3) Multiply the result by `scale`.

    Args:
        input (Tensor): shape [N, C, ...], or at least has dimension for channels.
        bias (Optional[Tensor]): shape [C], or None if no bias needed.
        negative_slope (float): Slope for the negative part of LeakyReLU.
        scale (float): Constant factor to scale the output by.

    Returns:
        Tensor of same shape as input.
    """
    # 1) Add bias if provided
    if bias is not None:
        # Broadcast bias across the non-channel dimensions
        # E.g. for 4D input [N,C,H,W], shape => [1, C, 1, 1]
        ndim = input.ndim
        shape = [1, -1] + [1] * (ndim - 2)
        input = input + bias.view(shape)

    # 2) Apply LeakyReLU
    #    negative_slope is typically 0.2 in StyleGAN, but can be changed
    out = F.leaky_relu(input, negative_slope=negative_slope)

    # 3) Multiply by scale (often sqrt(2) ~ 1.4142)
    out = out * scale

    return out


import torch
from torch import nn
import torch.nn.functional as F

class FusedLeakyReLU(nn.Module):
    """
    A pure-Python/PyTorch version of FusedLeakyReLU:
      1) Optionally add bias,
      2) LeakyReLU,
      3) multiply by scale.
    """
    def __init__(self, channels, bias=True, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        self.negative_slope = negative_slope
        self.scale = scale

        if bias:
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Add bias
        if self.bias is not None:
            ndim = x.ndim
            shape = [1, -1] + [1] * (ndim - 2)
            x = x + self.bias.view(shape)

        # 2) LeakyReLU
        x = F.leaky_relu(x, negative_slope=self.negative_slope)

        # 3) Scale
        x = x * self.scale

        return x

