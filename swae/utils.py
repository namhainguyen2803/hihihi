import numpy as np
import torch
import ot
from torch.nn.functional import pad
from swae.distributions import rand_cirlce2d, rand_ring2d, rand_uniform2d

def quantile_function(qs, cws, xs):
    n = xs.shape[0]
    cws = cws.T.contiguous()
    qs = qs.T.contiguous()
    idx = torch.searchsorted(cws, qs, right=False).T
    return torch.gather(xs, 0, torch.clamp(idx, 0, n - 1))
def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cpu',e=0):
    if(e==0):
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

def rand_projections(dim, num_projections=1000,device='cpu'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance

def one_dimensional_Wasserstein(X, Y,theta, u_weights=None, v_weights=None, p=2):
    if (X.shape[0] == Y.shape[0] and u_weights is None and v_weights is None):
        return one_dimensional_Wasserstein_prod(X,Y,theta,p)
    u_values = torch.matmul(X, theta.transpose(0, 1))
    v_values = torch.matmul(Y, theta.transpose(0, 1))
    n = u_values.shape[0]
    m = v_values.shape[0]
    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1. / n,
                               dtype=u_values.dtype, device=u_values.device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(
            u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = torch.full(v_values.shape, 1. / m,
                               dtype=v_values.dtype, device=v_values.device)
    elif v_weights.ndim != v_values.ndim:
        v_weights = torch.repeat_interleave(
            v_weights[..., None], v_values.shape[-1], -1)

    u_sorter = torch.sort(u_values, 0)[1]
    u_values = torch.gather(u_values, 0, u_sorter)

    v_sorter = torch.sort(v_values, 0)[1]
    v_values = torch.gather(v_values, 0, v_sorter)

    u_weights = torch.gather(u_weights, 0, u_sorter)
    v_weights = torch.gather(v_weights, 0, v_sorter)

    u_cumweights = torch.cumsum(u_weights, 0)
    v_cumweights = torch.cumsum(v_weights, 0)

    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 0), 0)[0]
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)

    pad_width = [(1, 0)] + (qs.ndim - 1) * [(0, 0)]
    how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
    qs = pad(qs, how_pad)

    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)
    return torch.sum(delta * torch.pow(diff_quantiles, p), dim=0)

def BSW(Xs,X,L=10,p=2,device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod,dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted-X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)# K\times L
    sw = torch.mean(wasserstein_distance,dim=1)
    return torch.mean(torch.pow(sw, 1. / p))

def BSW_list(Xs,X,L=10,p=2,device='cpu'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i],X,theta) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance,dim=0)
    sw = torch.mean(wasserstein_distance,dim=1)
    return torch.mean(torch.pow(sw, 1. / p))

def FBSW(Xs,X,L=10,p=2,device='cpu'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod,dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted-X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)# K\times L
    sw = torch.mean(wasserstein_distance,dim=1)
    return torch.max(torch.pow(sw, 1. / p))

def FBSW_list(Xs,X,L=10,p=2,device='cpu'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i], X, theta) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance, dim=0)
    sw = torch.mean(wasserstein_distance, dim=1)
    return torch.max(torch.pow(sw, 1. / p))

def FEBSW(Xs,X,L=10,p=2,device='cpu',reduce='max',energy='max'):
    dim = X.size(1)
    theta = rand_projections(dim, L, device)
    Xs_prod = torch.matmul(Xs, theta.transpose(0, 1))
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Xs_prod_sorted = torch.sort(Xs_prod,dim=1)[0]
    X_prod_sorted = torch.sort(X_prod, dim=0)[0]
    wasserstein_distance = torch.abs(Xs_prod_sorted-X_prod_sorted)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=1)# K\times L
    wasserstein_distanceT = wasserstein_distance.T.view(L,-1,1)
    M = torch.cdist(wasserstein_distanceT,wasserstein_distanceT,p=2)
    if (energy == 'max'):
        weights = torch.softmax(torch.max(torch.max(M, dim=1)[0], dim=1)[0].view(1, -1), dim=1)
    elif (energy == 'mean'):
        weights = torch.softmax(torch.mean(M, dim=[1, 2]).view(1, -1),
                                dim=1)  # weights = torch.softmax(torch.mean(M,dim=[1,2]).view(1,-1) + torch.mean(wasserstein_distance,dim=0,keepdim=True), dim=1)
    sws = torch.sum(weights * wasserstein_distance, dim=1)
    sws = torch.pow(sws, 1. / p)
    if(reduce=='max'):
        return torch.max(sws)
    elif(reduce=='mean'):
        return torch.mean(sws)

def FEBSW_list(Xs,X,L=10,p=2,device='cpu',reduce='max',energy='max'):
    dim = X.size(1)
    K = len(Xs)
    theta = rand_projections(dim, L, device)
    wasserstein_distance = [one_dimensional_Wasserstein(Xs[i], X, theta) for i in range(K)]
    wasserstein_distance = torch.stack(wasserstein_distance, dim=0)
    wasserstein_distanceT = wasserstein_distance.T.view(L,-1,1)
    M = torch.cdist(wasserstein_distanceT,wasserstein_distanceT,p=2)
    if (energy == 'max'):
        weights = torch.softmax(torch.max(torch.max(M, dim=1)[0], dim=1)[0].view(1, -1), dim=1)
    elif (energy == 'mean'):
        weights = torch.softmax(torch.mean(M, dim=[1, 2]).view(1, -1),
                                dim=1)  # weights = torch.softmax(torch.mean(M,dim=[1,2]).view(1,-1) + torch.mean(wasserstein_distance,dim=0,keepdim=True), dim=1)
    else:
        weights = torch.ones_like(wasserstein_distance)
    sws = torch.sum(weights * wasserstein_distance, dim=1)
    sws = torch.pow(sws, 1. / p)
    if(reduce=='max'):
        return torch.max(sws)
    elif(reduce=='mean'):
        return torch.mean(sws)

def rand_projection(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)

def ssliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2,
                                 device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projection(embedding_dim, num_projections).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()

def sliced_wasserstein_distance(encoded_samples,
                                distribution_fn=rand_cirlce2d,
                                num_projections=50,
                                p=2,
                                device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive batch size from encoded samples
    batch_size = encoded_samples.size(0)
    # draw random samples from latent space prior distribution
    z = distribution_fn(batch_size).to(device)
    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = ssliced_wasserstein_distance(encoded_samples, z,
                                       num_projections, p, device)
    return swd