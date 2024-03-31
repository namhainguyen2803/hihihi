import argparse
import os

from dataloader.dataloader import *
from eval_functions import *
from swae.distributions import rand_cirlce2d, rand_ring2d, rand_uniform2d, rand, randn
from swae.models.cifar10 import CIFAR10Autoencoder
from swae.models.mnist import MNISTAutoencoder
from swae.trainer import SWAEBatchTrainer
import torch.nn.functional as F


def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch')

    parser.add_argument('--pretrained-weight',
                        default='/kaggle/input/swae_fairsw/pytorch/fefbsw_200epochs/1/mnist_epoch_200.pth',
                        help='link of pretrained model, '
                             'e.g: /kaggle/input/swae_fairsw/pytorch/fefbsw_200epochs/1/mnist_epoch_200.pth')

    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')

    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')

    parser.add_argument('--weight_swd', type=float, default=1,
                        help='weight of swd (default: 1)')
    parser.add_argument('--weight_fsw', type=float, default=1,
                        help='weight of fsw (default: 1)')

    parser.add_argument('--method', type=str, default='FEFBSW', metavar='MED',
                        help='method (default: FEFBSW)')
    parser.add_argument('--num-projections', type=int, default=10000, metavar='NP',
                        help='number of projections (default: 500)')

    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer (default: adam)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')

    args = parser.parse_args()
    # create output directory
    imagesdir = os.path.join(args.outdir, 'images')
    chkptdir = os.path.join(args.outdir, 'models')
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(imagesdir, exist_ok=True)
    os.makedirs(chkptdir, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # build train and test set data loaders
    if args.dataset == 'mnist':
        data_loader = MNISTLTDataLoader(train_batch_size=args.batch_size, test_batch_size=args.batch_size)
    elif args.dataset == 'cifar10':
        data_loader = CIFAR10LTDataLoader(train_batch_size=args.batch_size, test_batch_size=args.batch_size)
    else:
        data_loader = None
    train_loader, test_loader = data_loader.create_dataloader()

    # create encoder and decoder
    if args.dataset == 'mnist':
        model = MNISTAutoencoder().to(device)
    elif args.dataset == 'cifar10':
        model = CIFAR10Autoencoder(embedding_dim=args.embedding_size).to(device)
    else:
        model = None

    if args.dataset == 'mnist':
        if args.distribution == 'circle':
            distribution_fn = rand_cirlce2d
        elif args.distribution == 'ring':
            distribution_fn = rand_ring2d
        else:
            distribution_fn = rand_uniform2d
    else:
        if args.distribution == 'uniform':
            distribution_fn = rand(args.embedding_size)
        elif args.distribution == 'normal':
            distribution_fn = randn(args.embedding_size)
        else:
            raise ('distribution {} not supported'.format(args.distribution))

    print(model)

    model.load_state_dict(torch.load(args.pretrained_weight))

    evaluator = SWAEBatchTrainer(autoencoder=model,
                                 optimizer=None,
                                 distribution_fn=distribution_fn,
                                 num_classes=data_loader.num_classes,
                                 num_projections=args.num_projections,
                                 weight_swd=args.weight_swd,
                                 weight_fsw=args.weight_fsw,
                                 device=device, method=args.method)

    with torch.no_grad():

        ultimate_evaluation(args=args,
                            model=model,
                            evaluator=evaluator,
                            test_loader=test_loader,
                            prior_distribution=distribution_fn,
                            theta=None,
                            theta_latent=None,
                            device=device)

if __name__ == '__main__':
    main()
