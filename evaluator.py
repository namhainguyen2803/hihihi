import argparse
import matplotlib as mpl

mpl.use('Agg')
from sklearn.manifold import TSNE
import torch
from swae.models.cifar10 import CIFAR10Autoencoder
from swae.models.mnist import MNISTAutoencoder
from swae.trainer import SWAEBatchTrainer
from swae.distributions import rand_cirlce2d, rand_ring2d, rand_uniform2d, rand, randn

from evaluate.eval_ws import *
from evaluate.eval_fid import *

import torch.optim as optim
import torchvision.utils as vutils
from dataloader.dataloader import *
from utils import *


def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch')
    parser.add_argument('--dataset', default='cifar10', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')
    parser.add_argument('--images-path', default='/images/', help='path to images')

    parser.add_argument('--batch-size', type=int, default=500, metavar='BS',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--batch-size-test', type=int, default=500, metavar='BST',
                        help='input batch size for evaluating (default: 500)')

    parser.add_argument('--pretrained-model', type=str, metavar='S',
                        help='pretrained model path')
    parser.add_argument('--method', type=str, default='EFBSW', metavar='MED',
                        help='method (default: EFBSW)')
    parser.add_argument('--num-projections', type=int, default=10000, metavar='NP',
                        help='number of projections (default: 500)')
    parser.add_argument('--embedding-size', type=int, default=48, metavar='ES',
                        help='embedding latent space (default: 48)')

    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')
    parser.add_argument('--stat-dir', default='stats', help='path to images')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    os.makedirs(args.images_path, exist_ok=True)
    args.images_path = os.path.join(args.images_path, args.dataset)
    gen_dir = os.path.join(args.images_path, args.method)
    os.makedirs(gen_dir, exist_ok=True)
    args.gen_dir = gen_dir

    args.stat_dir = os.path.join(args.stat_dir, args.dataset)

    print(f"gen_dir: {args.gen_dir}")
    print(f"stat_dir: {args.stat_dir}")

    if args.dataset == 'mnist':
        data_loader = MNISTLTDataLoader(train_batch_size=args.batch_size, test_batch_size=args.batch_size_test)
    elif args.dataset == 'cifar10':
        data_loader = CIFAR10LTDataLoader(data_dir=args.datadir, train_batch_size=args.batch_size,
                                          test_batch_size=args.batch_size_test)
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
    print(model)

    model.load_state_dict(torch.load(args.pretrained_model))

    # determine latent distribution
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

    with torch.no_grad():
        RL, LP, WG, F, AD, F_images, AD_images = ultimate_evaluate_fid(args=args,
                                                                       model=model,
                                                                       test_loader=test_loader,
                                                                       prior_distribution=distribution_fn,
                                                                       device=device)

        output_file = f'{args.outdir}/output_{args.method}.log'
        with open(output_file, 'a') as f:
            f.write("In testing, when evaluating in train loader:\n")
            f.write(f" +) Reconstruction loss (RL): {RL}\n")
            f.write(f" +) Wasserstein distance between generated and real images (WG): {WG}\n")
            f.write(f" +) Wasserstein distance between posterior and prior distribution (LP): {LP}\n")
            f.write(f" +) Fairness (F): {F}\n")
            f.write(f" +) Averaging distance (AD): {AD}\n")
            f.write(f" +) Fairness in images space (FI): {F_images}\n")
            f.write(f" +) Averaging distance in images space (ADI): {AD_images}\n")
            f.write("\n")

if __name__ == "__main__":
    main()