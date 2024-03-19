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
                 num_projections=50, p=2, weight_swd=3, weight_fsw=1, device=None):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_.encoder.embedding_dim_
        self.num_projections_ = num_projections
        self.p_ = p
        self._device = device if device else torch.device('cpu')
        self.num_classes = num_classes

        self.weight = weight_swd
        self.weight_fsw = weight_fsw

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
        with torch.no_grad():
            x = x.to(self._device)
            y = y.to(self._device)
            recon_x, z_posterior = self.model_(x)

            batch_size = x.size(0)

            list_z = list()

            list_recon_loss = dict()
            list_swd = dict()
            list_l1 = dict()

            l1_loss, bce_loss, swd_loss = 0, 0, 0

            for cls in range(self.num_classes):
                recon_x_cls = recon_x[y == cls]
                x_cls = x[y == cls]

                z_cls = z_posterior[y == cls]
                list_z.append(z_cls)

                z_sample = self._distribution_fn(z_cls.shape[0]).to(self._device)

                recon_loss = F.binary_cross_entropy(recon_x_cls, x_cls)
                latent_ws_dist = sliced_wasserstein_distance(encoded_samples=z_cls, distribution_samples=z_sample,
                                                             num_projections=self.num_projections_, p=self.p_,
                                                             device=self._device)

                l1 = F.l1_loss(recon_x_cls, x_cls)

                list_swd[cls] = latent_ws_dist
                list_recon_loss[cls] = recon_loss
                list_l1[cls] = l1

                bce_loss += recon_loss
                l1_loss += l1
                swd_loss += latent_ws_dist

            bce_loss /= batch_size
            l1_loss /= batch_size
            swd_loss /= batch_size

            z_prior = self._distribution_fn(batch_size).to(self._device)
            fsw = FEFBSW_list(Xs=list_z, X=z_prior, device=self._device)

            loss = bce_loss + float(self.weight_fsw) * fsw + float(self.weight) * swd_loss + l1_loss

            return {
                'list_recon': list_recon_loss,
                'list_swd': list_swd,
                'list_l1': list_l1,

                'loss': loss,
                'recon_loss':bce_loss,
                'swd_loss': swd_loss,
                'l1_loss': l1_loss,
                'fsw_loss': fsw,

                'encode': z_posterior,
                'decode': recon_x
            }

    def eval_on_batch(self, x, y):

        x = x.to(self._device)
        y = y.to(self._device)

        recon_x, z_posterior = self.model_(x)
        bce = F.binary_cross_entropy(recon_x, x)

        l1 = F.l1_loss(recon_x, x)

        batch_size = x.size(0)
        z_prior = self._distribution_fn(batch_size).to(self._device)

        swd = sliced_wasserstein_distance(encoded_samples=z_posterior, distribution_samples=z_prior,
                                          num_projections=self.num_projections_, p=self.p_,
                                          device=self._device)

        list_z_posterior = list()
        for cls in range(self.num_classes):
            list_z_posterior.append(z_posterior[y == cls])

        fsw = FEFBSW_list(Xs=list_z_posterior, X=z_prior, device=self._device)

        loss = bce + float(self.weight_fsw) * fsw + float(self.weight) * swd + l1

        return {
            'loss': loss,
            'bce': bce,
            'fsw_loss': fsw,
            'w2': swd,
            'encode': z_posterior,
            'decode': recon_x,
            'l1': l1
        }
