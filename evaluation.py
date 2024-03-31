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

    evaluator = SWAEBatchTrainer(autoencoder=model, optimizer=None,
                                 distribution_fn=distribution_fn,
                                 num_classes=data_loader.num_classes,
                                 num_projections=args.num_projections,
                                 weight_swd=args.weight_swd, weight_fsw=args.weight_fsw,
                                 device=device, method=args.method)

    with torch.no_grad():

        list_real_images = list()
        list_labels = list()
        list_encoded_images = list()
        list_decoded_images = list()

        for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
            list_real_images.append(x_test)
            list_labels.append(y_test)

            test_evals = evaluator.forward(x_test)

            list_encoded_images.append(test_evals["encode"].detach())
            list_decoded_images.append(test_evals["decode"].detach())

        tensor_real_images = torch.cat(list_real_images, dim=0).cpu()
        tensor_labels = torch.cat(list_labels, dim=0).cpu()
        tensor_encoded_images = torch.cat(list_encoded_images, dim=0).cpu()
        tensor_decoded_images = torch.cat(list_decoded_images, dim=0).cpu()
        tensor_generated_images = generate_image(model=model,
                                                 prior_distribution=distribution_fn,
                                                 num_images=tensor_real_images.shape[0],
                                                 device=device)

        num_images = tensor_real_images.shape[0]
        print(num_images)

        tensor_flatten_real_images = tensor_real_images.view(num_images, -1)
        tensor_flatten_encoded_images = tensor_encoded_images.view(num_images, -1)
        tensor_flatten_decoded_images = tensor_decoded_images.view(num_images, -1)
        tensor_flatten_generated_images = tensor_generated_images.view(num_images, -1)

        RL = torch.nn.functional.binary_cross_entropy(tensor_decoded_images, tensor_real_images)

        print(f"Reconstruction loss: {RL}")

        theta = rand_projections(dim=tensor_flatten_real_images.shape[-1],
                                 num_projections=args.num_projections,
                                 device=device)

        WG = sliced_wasserstein_distance(encoded_samples=tensor_generated_images,
                                         distribution_samples=tensor_real_images,
                                         num_projections=args.num_projections,
                                         p=2,
                                         device=device,
                                         theta=theta)

        print(f"Wasserstein distance between generated and real images: {WG}")

        prior_samples = distribution_fn(num_images).to(device)
        theta_latent = rand_projections(dim=tensor_encoded_images.shape[-1],
                                        num_projections=args.num_projections,
                                        device=device)
        LP = sliced_wasserstein_distance(encoded_samples=tensor_encoded_images,
                                         distribution_samples=prior_samples,
                                         num_projections=args.num_projections,
                                         p=2,
                                         device=device,
                                         theta=theta_latent)

        print(f"Wasserstein distance between posterior and prior distribution: {LP}")

        F, AD = compute_fairness_and_averaging_distance(list_features=tensor_flatten_encoded_images,
                                                        list_labels=tensor_labels,
                                                        prior_distribution=distribution_fn,
                                                        num_classes=data_loader.num_classes,
                                                        device=device,
                                                        num_projections=args.num_projections,
                                                        dim=tensor_encoded_images.shape[-1],
                                                        theta=theta)

        print(f"Fairness: {F}")
        print(f"Averaging distance: {AD}")

        F_images, AD_images = compute_fairness_and_averaging_distance_in_images_space(model=model,
                                                                                      prior_distribution=distribution_fn,
                                                                                      test_loader=test_loader,
                                                                                      device=device,
                                                                                      num_projections=args.num_projections,
                                                                                      theta=theta)
        print(f"Fairness in images space: {F_images}")
        print(f"Averaging distance in images space: {AD_images}")


if __name__ == '__main__':
    main()
