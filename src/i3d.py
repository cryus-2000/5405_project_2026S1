"""Minimal Inception-I3D implementation used for RGB feature extraction.

This file is intentionally self-contained so the project does not depend on an
external I3D package at runtime. The retrieval code uses features from the
`Mixed_5c` endpoint, then average-pools them into one vector per snippet.
"""

import torch
from torch import nn
import torch.nn.functional as F


def same_padding(input_size, kernel_size, stride):
    if input_size % stride == 0:
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - (input_size % stride), 0)
    return pad


class MaxPool3dSamePadding(nn.MaxPool3d):
    def forward(self, x):
        batch, channel, time, height, width = x.size()
        kernel_t, kernel_h, kernel_w = self.kernel_size
        stride_t, stride_h, stride_w = self.stride

        pad_t = same_padding(time, kernel_t, stride_t)
        pad_h = same_padding(height, kernel_h, stride_h)
        pad_w = same_padding(width, kernel_w, stride_w)

        pad_front = pad_t // 2
        pad_back = pad_t - pad_front
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        return super().forward(x)


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
    ):
        super().__init__()
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._kernel_shape = kernel_shape
        self._stride = stride
        self.name = name

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        batch, channel, time, height, width = x.size()
        kernel_t, kernel_h, kernel_w = self._kernel_shape
        stride_t, stride_h, stride_w = self._stride

        pad_t = same_padding(time, kernel_t, stride_t)
        pad_h = same_padding(height, kernel_h, stride_h)
        pad_w = same_padding(width, kernel_w, stride_w)

        pad_front = pad_t // 2
        pad_back = pad_t - pad_front
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()
        out_0, out_1a, out_1b, out_2a, out_2b, out_3b = out_channels
        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_0,
            kernel_shape=(1, 1, 1),
            name=f"{name}/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_1a,
            kernel_shape=(1, 1, 1),
            name=f"{name}/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_1a,
            output_channels=out_1b,
            kernel_shape=(3, 3, 3),
            name=f"{name}/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_2a,
            kernel_shape=(1, 1, 1),
            name=f"{name}/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_2a,
            output_channels=out_2b,
            kernel_shape=(3, 3, 3),
            name=f"{name}/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_3b,
            kernel_shape=(1, 1, 1),
            name=f"{name}/Branch_3/Conv3d_0b_1x1",
        )

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3D(nn.Module):
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(self, num_classes=400, in_channels=3, final_endpoint="Logits"):
        super().__init__()
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")
        self._num_classes = num_classes
        self._final_endpoint = final_endpoint

        self.end_points = {}
        self._add_endpoint(
            "Conv3d_1a_7x7",
            Unit3D(
                in_channels=in_channels,
                output_channels=64,
                kernel_shape=(7, 7, 7),
                stride=(2, 2, 2),
                name="inception_i3d/Conv3d_1a_7x7",
            ),
        )
        self._add_endpoint(
            "MaxPool3d_2a_3x3",
            MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0),
        )
        self._add_endpoint(
            "Conv3d_2b_1x1",
            Unit3D(64, 64, kernel_shape=(1, 1, 1), name="inception_i3d/Conv3d_2b_1x1"),
        )
        self._add_endpoint(
            "Conv3d_2c_3x3",
            Unit3D(64, 192, kernel_shape=(3, 3, 3), name="inception_i3d/Conv3d_2c_3x3"),
        )
        self._add_endpoint(
            "MaxPool3d_3a_3x3",
            MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0),
        )
        self._add_endpoint("Mixed_3b", InceptionModule(192, [64, 96, 128, 16, 32, 32], name="inception_i3d/Mixed_3b"))
        self._add_endpoint("Mixed_3c", InceptionModule(256, [128, 128, 192, 32, 96, 64], name="inception_i3d/Mixed_3c"))
        self._add_endpoint(
            "MaxPool3d_4a_3x3",
            MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0),
        )
        self._add_endpoint("Mixed_4b", InceptionModule(480, [192, 96, 208, 16, 48, 64], name="inception_i3d/Mixed_4b"))
        self._add_endpoint("Mixed_4c", InceptionModule(512, [160, 112, 224, 24, 64, 64], name="inception_i3d/Mixed_4c"))
        self._add_endpoint("Mixed_4d", InceptionModule(512, [128, 128, 256, 24, 64, 64], name="inception_i3d/Mixed_4d"))
        self._add_endpoint("Mixed_4e", InceptionModule(512, [112, 144, 288, 32, 64, 64], name="inception_i3d/Mixed_4e"))
        self._add_endpoint("Mixed_4f", InceptionModule(528, [256, 160, 320, 32, 128, 128], name="inception_i3d/Mixed_4f"))
        self._add_endpoint(
            "MaxPool3d_5a_2x2",
            MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
        )
        self._add_endpoint("Mixed_5b", InceptionModule(832, [256, 160, 320, 32, 128, 128], name="inception_i3d/Mixed_5b"))
        self._add_endpoint("Mixed_5c", InceptionModule(832, [384, 192, 384, 48, 128, 128], name="inception_i3d/Mixed_5c"))

        if self._has_endpoint("Logits"):
            self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
            self.dropout = nn.Dropout(0.5)
            self.logits = Unit3D(
                in_channels=1024,
                output_channels=self._num_classes,
                kernel_shape=(1, 1, 1),
                activation_fn=None,
                use_batch_norm=False,
                use_bias=True,
                name="inception_i3d/Logits/Conv3d_0c_1x1",
            )

    def _has_endpoint(self, endpoint):
        return self.VALID_ENDPOINTS.index(endpoint) <= self.VALID_ENDPOINTS.index(self._final_endpoint)

    def _add_endpoint(self, name, module):
        if self._has_endpoint(name):
            self.end_points[name] = module
            self.add_module(name, module)

    def extract_features(self, x):
        for endpoint in self.VALID_ENDPOINTS:
            if endpoint in self.end_points:
                x = self._modules[endpoint](x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        if self._final_endpoint == "Mixed_5c":
            return x
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._final_endpoint == "Logits":
            return x
        return torch.mean(x, dim=[2, 3, 4])


def clean_i3d_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        key = key.removeprefix("model.")
        if key.startswith("logits.") or key.startswith("Logits."):
            continue
        cleaned[key] = value
    return cleaned
