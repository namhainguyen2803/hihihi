import numpy as np

import torch
import torch.nn.functional as F
from swae.utils import *
from swae.distributions import rand_cirlce2d
from swae.utils import *


class SWAEBatchTrainer:
    """ Sliced Wasserstein Autoencoder Batch Trainer.

        Args:
            autoencoder (torch.nn.Module): module which implements autoencoder framework
            optimizer (torch.optim.Optimizer): torch optimizer
            distribution_fn (callable): callable to draw random samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            weight (float): weight of divergence metric compared to reconstruction in loss
            device (torch.Device): torch device
    """

    def __init__(self, autoencoder, optimizer, distribution_fn, num_classes=10,
                 num_projections=50, p=2, weight=10, device=None):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_.encoder.embedding_dim_
        self.num_projections_ = num_projections
        self.p_ = p
        self.weight = weight
        self._device = device if device else torch.device('cpu')
        self.num_classes = num_classes

    def __call__(self, x):
        return self.eval_on_batch(x)

    def train_on_batch(self, x):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x)
        # backpropagate loss
        evals['loss'].backward()
        # update encoder and decoder parameters
        self.optimizer.step()
        return evals

    def test_on_batch(self, x):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x)
        return evals

    def eval_on_batch(self, x):
        x = x.to(self._device)
        recon_x, z = self.model_(x)
        # mutual information reconstruction loss
        bce = F.cross_entropy(recon_x, x)
        # for explaination of additional L1 loss see references in README.md
        # high lvl summary prevents variance collapse on latent variables
        l1 = F.l1_loss(recon_x, x)
        # divergence on transformation plane from X space to Z space to match prior
        _swd = sliced_wasserstein_distance(z, self._distribution_fn,
                                           self.num_projections_, self.p_,
                                           self._device)
        w2 = float(self.weight) * _swd  # approximate wasserstein-2 distance
        loss = bce + l1 + w2
        return {
            'loss': loss,
            'bce': bce,
            'l1': l1,
            'w2': w2,
            'encode': z,
            'decode': recon_x
        }
