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
        recon_x, z_posterior = self.model_(x)
        print(x)
        bce = F.binary_cross_entropy(recon_x, x)

        l1 = F.l1_loss(recon_x, x)

        batch_size = x.size(0)
        z_prior = self._distribution_fn(batch_size).to(self._device)

        swd = sliced_wasserstein_distance(encoded_samples=z_posterior, distribution_samples=z_prior,
                                          num_projections=self.num_projections_, p=self.p_,
                                          device=self._device)

        loss = bce + float(self.weight) * swd + l1
        return {
            'loss': loss,
            'bce': bce,
            'l1': l1,
            'w2': swd,
            'encode': z_posterior,
            'decode': recon_x
        }
