import numpy as np

import torch
import torch.nn.functional as F
from swae.utils import *
from swae.distributions import rand_cirlce2d



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
                 num_projections=50, p=2, weight=10.0, device=None):
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

    def train_on_batch(self, x, y):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x, y)
        # backpropagate loss
        evals['loss'].backward()
        # update encoder and decoder parameters
        self.optimizer.step()
        return evals

    def test_on_batch(self, x, y):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        return self.eval_on_batch(x, y)

    def eval_on_batch(self, x, y):
        x = x.to(self._device)
        y = y.to(self._device)
        recon_x, z = self.model_(x)
        bce = F.cross_entropy(recon_x, x)
        _swd = sliced_wasserstein_distance(z, self._distribution_fn,
                                           self.num_projections_, self.p_,
                                           self._device)
        list_z = list()

        batch_size = x.size(0)
        z = self._distribution_fn(batch_size).to(self._device)

        for cls in range(self.num_classes):
            list_z.append(z[y == cls])

        fsw = FEBSW_list(Xs=list_z, X=z, device=self._device)
        w2 = float(self.weight) * _swd
        loss = bce + fsw + w2

        return {
            'loss': loss,
            'bce': bce,
            'FairSW': fsw,
            'w2': w2,
            'encode': z,
            'decode': recon_x
        }
