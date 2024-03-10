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
                 num_projections=50, p=2, weight=5, device=None):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_.encoder.embedding_dim_
        self.num_projections_ = num_projections
        self.p_ = p
        self.weight = weight
        self._device = device if device else torch.device('cpu')
        self.num_classes = num_classes

        self.weight_fsw = 5

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
        with torch.no_grad:

            x = x.to(self._device)
            y = y.to(self._device)
            recon_x, z_posterior = self.model_(x)
            bce = F.cross_entropy(recon_x, x)

            batch_size = x.size(0)
            z_prior = self._distribution_fn(batch_size).to(self._device)

            _swd = sliced_wasserstein_distance(encoded_samples=z_posterior, distribution_samples=z_prior,
                                               num_projections=self.num_projections_, p=self.p_,
                                               device=self._device)
            w2 = float(self.weight) * _swd

            list_z = list()
            wasserstein_distances = dict()
            for cls in range(self.num_classes):
                z_cls = z_posterior[y == cls]
                list_z.append(z_cls)
                z_sample = self._distribution_fn(z_cls.shape[0]).to(self._device)
                ws_dist = sliced_wasserstein_distance(encoded_samples=z_cls, distribution_samples=z_sample,
                                                     num_projections=self.num_projections_, p=self.p_,
                                                     device=self._device)
                wasserstein_distances[cls] = ws_dist

            fsw = FEBSW_list(Xs=list_z, X=z_prior, device=self._device)

            loss = bce + fsw + w2

            return {
                'loss': loss,
                'bce': bce,
                'FairSW': fsw,
                'w2': w2,
                'encode': z_posterior,
                'decode': recon_x,
                'wasserstein_distances': wasserstein_distances
            }

    def eval_on_batch(self, x, y):

        x = x.to(self._device)
        y = y.to(self._device)

        recon_x, z_posterior = self.model_(x)
        bce = F.cross_entropy(recon_x, x)

        batch_size = x.size(0)
        z_prior = self._distribution_fn(batch_size).to(self._device)

        swd = sliced_wasserstein_distance(encoded_samples=z_posterior, distribution_samples=z_prior,
                                           num_projections=self.num_projections_, p=self.p_,
                                           device=self._device)

        list_z_posterior = list()
        for cls in range(self.num_classes):
            list_z_posterior.append(z_posterior[y == cls])

        fsw = FEBSW_list(Xs=list_z_posterior, X=z_prior, device=self._device)

        loss = bce + float(self.weight_fsw) * fsw + float(self.weight) * swd

        return {
            'loss': loss,
            'bce': bce,
            'FairSW': fsw,
            'w2': swd,
            'encode': z_posterior,
            'decode': recon_x
        }
