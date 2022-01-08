import random
from typing import (
    Iterable,
    List,
    Optional,
)

import numpy as np
import torch
import torch.nn as nn

from .vgg import VGG19
from .. import functional as F
from ..config import LOSS_TYPES


class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(
            self,
            band_width: float = 0.5,
            loss_type: str = 'cosine',
            use_vgg: bool = False,
            vgg_model: nn.Module = None,
            vgg_layers: List[str] = ['relu3_4'],
            feature_1d_size: int = 64,
    ):

        super().__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert loss_type in LOSS_TYPES,\
            f'select a loss type from {LOSS_TYPES}.'

        self.loss_type = loss_type
        self.band_width = band_width
        self.feature_1d_size = feature_1d_size

        if use_vgg:
            self.vgg_model = VGG19() if vgg_model is None else vgg_model
            self.vgg_layers = vgg_layers
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor, all_dist: bool = False):
        if not hasattr(self, 'vgg_model'):
            return self.contextual_loss(x, y, self.feature_1d_size, self.band_width, all_dist=all_dist)


        x = self.forward_vgg(x)
        y = self.forward_vgg(y)

        loss = 0
        for layer in self.vgg_layers:
            # picking up vgg feature maps
            fx = getattr(x, layer)
            fy = getattr(y, layer)
            loss = loss + self.contextual_loss(
                fx, fy, self.feature_1d_size, self.band_width, all_dist=all_dist, loss_type=self.loss_type
            )
        return loss

    def forward_vgg(self, x: torch.Tensor):
        assert x.shape[1] == 3, 'VGG model takes 3 chennel images.'
        # [-1, 1] -> [0, 1]
        x = (x + 1) * 0.5

        # normalization
        x = x.sub(self.vgg_mean.detach()).div(self.vgg_std)
        return self.vgg_model(x)

    @classmethod
    def contextual_loss(
            cls,
            x: torch.Tensor, y: torch.Tensor,
            feature_1d_size: int,
            band_width: int,
            all_dist: bool = False,
            loss_type: str = 'cosine',
    ) -> torch.Tensor:
        feature_size = feature_1d_size ** 2
        if np.prod(x.shape[2:]) > feature_size or np.prod(y.shape[2:]) > feature_size:
            x, indices = cls.random_sampling(x, feature_1d_size=feature_1d_size)
            y, _ = cls.random_sampling(y, feature_1d_size=feature_1d_size, indices=indices)

        return F.contextual_loss(x, y, band_width, all_dist=all_dist, loss_type=loss_type)

    @staticmethod
    def random_sampling(
            tensor_NCHW: torch.Tensor, feature_1d_size: int, indices: Optional[List] = None
    ):
        N, C, H, W = tensor_NCHW.shape
        S = H * W
        tensor_NCS = tensor_NCHW.reshape([N, C, S])
        if indices is None:
            all_indices = list(range(S))
            random.shuffle(all_indices)
            indices = all_indices[:feature_1d_size**2]
        res = tensor_NCS[:, :, indices].reshape(N, -1, feature_1d_size, feature_1d_size)
        return res, indices
