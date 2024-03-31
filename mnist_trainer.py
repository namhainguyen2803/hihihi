import argparse
import os

import matplotlib as mpl
from sklearn.manifold import TSNE

from swae.models.cifar10 import CIFAR10Autoencoder

mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.utils as vutils
from swae.distributions import rand_cirlce2d, rand_ring2d, rand_uniform2d, rand, randn
from swae.models.mnist import MNISTAutoencoder
from swae.trainer import SWAEBatchTrainer
from torchvision import datasets, transforms
from dataloader.dataloader import *
from swae.utils import *
from eval_functions import *

def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    parser.add_argument('--weight_swd', type=float, default=1,
                        help='weight of swd (default: 1)')
    parser.add_argument('--weight_fsw', type=float, default=1,
                        help='weight of fsw (default: 1)')
    parser.add_argument('--method', type=str, default='FEFBSW', metavar='MED',
                        help='method (default: FEFBSW)')
    parser.add_argument('--num-projections', type=int, default=100000, metavar='NP',
                        help='number of projections (default: 500)')

    parser.add_argument('--alpha', type=float, default=0.9, metavar='A',
                        help='RMSprop alpha/rho (default: 0.9)')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='B1',
                        help='Adam beta1 (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam beta2 (default: 0.999)')

    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer (default: adam)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches to log training status (default: 10)')
    parser.add_argument('--log-epoch-interval', type=int, default=2, metavar='N',
                        help='number of epochs to save training artifacts (default: 1)')
    args = parser.parse_args()
    # create output directory
    imagesdir = os.path.join(args.outdir, 'images')
    chkptdir = os.path.join(args.outdir, 'models')
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(imagesdir, exist_ok=True)
    os.makedirs(chkptdir, exist_ok=True)
    # determine device and device dep. args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {'num_workers': args.num_workers,
                                                                                 'pin_memory': False}
    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    # log args
    if args.optimizer == 'rmsprop':
        print(
            'batch size {}\nepochs {}\nRMSprop lr {} alpha {}\ndistribution {}\nusing device {}\nseed set to {}'.format(
                args.batch_size, args.epochs, args.lr, args.alpha, args.distribution, device.type, args.seed
            ))
    else:
        print(
            'batch size {}\nepochs {}\n{}: lr {} betas {}/{}\ndistribution {}\nusing device {}\nseed set to {}'.format(
                args.batch_size, args.epochs, args.optimizer,
                args.lr, args.beta1, args.beta2, args.distribution,
                device.type, args.seed
            ))

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
    print(model)

    # create optimizer
    if args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

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

    # create batch sliced_wasserstein autoencoder trainer
    trainer = SWAEBatchTrainer(autoencoder=model, optimizer=optimizer,
                               distribution_fn=distribution_fn,
                               num_classes=data_loader.num_classes,
                               num_projections=args.num_projections,
                               weight_swd=args.weight_swd, weight_fsw=args.weight_fsw,
                               device=device, method=args.method)

    list_RL = list()
    list_LP = list()
    list_WG = list()
    list_F = list()
    list_AD = list()
    list_F_images = list()
    list_AD_images = list()

    list_loss = list()
    train_list_loss = list()

    with torch.no_grad():
        model.eval()

        RL, LP, WG, F, AD, F_images, AD_images = ultimate_evaluation(args=args,
                                                                     model=model,
                                                                     evaluator=trainer,
                                                                     test_loader=test_loader,
                                                                     prior_distribution=distribution_fn,
                                                                     theta=None,
                                                                     theta_latent=None,
                                                                     device=device)
        print("In pre-training, when evaluating test loader:")
        print(f"Reconstruction loss (RL): {RL}")
        print(f"Wasserstein distance between generated and real images (WG): {WG}")
        print(f"Wasserstein distance between posterior and prior distribution (LP): {LP}")
        print(f"Fairness (F): {F}")
        print(f"Averaging distance (AD): {AD}")
        print(f"Fairness in images space (FI): {F_images}")
        print(f"Averaging distance in images space (ADI): {AD_images}")
        print()

        list_RL.append(RL)
        list_WG.append(WG)
        list_LP.append(LP)
        list_F.append(F)
        list_AD.append(AD)
        list_AD_images.append(AD_images)
        list_F_images.append(F_images)


    print()
    # train networks for n epochs
    for epoch in range(args.epochs):
        print('training...')
        model.train()
        # if epoch > 10:
        #     trainer.weight *= 1.1
        # train autoencoder on train dataset

        for batch_idx, (x, y) in enumerate(train_loader, start=0):
            batch = trainer.train_on_batch(x, y)

            if (batch_idx + 1) % args.log_interval == 0:
                print('Train Epoch: {} ({:.2f}%) [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                    (batch_idx + 1), len(train_loader),
                    batch['loss'].item()))

        print('evaluating...')
        model.eval()
        with torch.no_grad():

            train_encode, train_targets, train_loss = list(), list(), 0.0
            test_encode, test_targets, test_loss = list(), list(), 0.0

            for test_batch_idx, (x, y) in enumerate(train_loader, start=0):
                batch = trainer.test_on_batch(x, y)

                train_encode.append(batch['encode'].detach())
                train_loss += batch['loss'].item()
                train_targets.append(y)

            for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
                test_evals = trainer.test_on_batch(x_test, y_test)

                test_encode.append(test_evals['encode'].detach())
                test_loss += test_evals['loss'].item()
                test_targets.append(y_test)

            test_loss /= len(test_loader)
            train_loss /= len(train_loader)
            list_loss.append(test_loss)
            train_list_loss.append(train_loss)

            print('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}'.format(epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                                                                  test_loss))
            print('{{"metric": "loss", "value": {}}}'.format(test_loss))

            RL, LP, WG, F, AD, F_images, AD_images = ultimate_evaluation(args=args,
                                                                         model=model,
                                                                         evaluator=trainer,
                                                                         test_loader=train_loader,
                                                                         prior_distribution=distribution_fn,
                                                                         theta=None,
                                                                         theta_latent=None,
                                                                         device=device)
            print("In pre-training, when evaluating train loader:")
            print(f"Reconstruction loss (RL): {RL}")
            print(f"Wasserstein distance between generated and real images (WG): {WG}")
            print(f"Wasserstein distance between posterior and prior distribution (LP): {LP}")
            print(f"Fairness (F): {F}")
            print(f"Averaging distance (AD): {AD}")
            print(f"Fairness in images space (FI): {F_images}")
            print(f"Averaging distance in images space (ADI): {AD_images}")
            print()

            RL, LP, WG, F, AD, F_images, AD_images = ultimate_evaluation(args=args,
                                                                         model=model,
                                                                         evaluator=trainer,
                                                                         test_loader=test_loader,
                                                                         prior_distribution=distribution_fn,
                                                                         theta=None,
                                                                         theta_latent=None,
                                                                         device=device)
            print("In pre-training, when evaluating test loader:")
            print(f"Reconstruction loss (RL): {RL}")
            print(f"Wasserstein distance between generated and real images (WG): {WG}")
            print(f"Wasserstein distance between posterior and prior distribution (LP): {LP}")
            print(f"Fairness (F): {F}")
            print(f"Averaging distance (AD): {AD}")
            print(f"Fairness in images space (FI): {F_images}")
            print(f"Averaging distance in images space (ADI): {AD_images}")
            print()

            list_RL.append(RL)
            list_WG.append(WG)
            list_LP.append(LP)
            list_F.append(F)
            list_AD.append(AD)
            list_AD_images.append(AD_images)
            list_F_images.append(F_images)

            if (epoch + 1) == args.epochs:
                # save model
                torch.save(model.state_dict(), '{}/{}_epoch_{}.pth'.format(chkptdir, args.dataset, epoch + 1))
                test_encode, test_targets = test_encode.cpu().numpy(), test_targets.cpu().numpy()
                train_encode, train_targets = train_encode.cpu().numpy(), train_targets.cpu().numpy()
                if args.dataset == "mnist":
                    # plot
                    plt.figure(figsize=(10, 10))
                    plt.scatter(test_encode[:, 0], -test_encode[:, 1], c=(10 * test_targets), cmap=plt.cm.Spectral)
                    plt.xlim([-1.5, 1.5])
                    plt.ylim([-1.5, 1.5])
                    plt.title('Test Latent Space\nLoss: {:.5f}'.format(test_loss))
                    plt.savefig('{}/test_latent_epoch_{}.png'.format(imagesdir, epoch + 1))
                    plt.close()

                    plt.figure(figsize=(10, 10))
                    plt.scatter(train_encode[:, 0], -train_encode[:, 1], c=(10 * train_targets), cmap=plt.cm.Spectral)
                    plt.xlim([-1.5, 1.5])
                    plt.ylim([-1.5, 1.5])
                    plt.title('Train Latent Space\nLoss: {:.5f}'.format(test_loss))
                    plt.savefig('{}/train_latent_epoch_{}.png'.format(imagesdir, epoch + 1))
                    plt.close()

                else:
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_result = tsne.fit_transform(test_encode)

                    plt.figure(figsize=(10, 10))
                    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=test_targets, cmap='viridis')
                    plt.title('t-SNE Visualization')
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.title('Test Latent Space\nLoss: {:.5f}'.format(test_loss))
                    plt.savefig('{}/test_latent_epoch_{}.png'.format(imagesdir, epoch + 1))
                    plt.colorbar(label='Target')
                    plt.close()

                # save sample input and reconstruction
                vutils.save_image(x_test,
                                  '{}/{}_test_samples_epoch_{}.png'.format(imagesdir, args.distribution, epoch + 1))

                vutils.save_image(test_evals['decode'].detach(),
                                  '{}/{}_test_recon_epoch_{}.png'.format(imagesdir, args.distribution, epoch + 1),
                                  normalize=True)

                vutils.save_image(x, '{}/{}_train_samples_epoch_{}.png'.format(imagesdir, args.distribution, epoch + 1))

                vutils.save_image(batch['decode'].detach(),
                                  '{}/{}_train_recon_epoch_{}.png'.format(imagesdir, args.distribution, epoch + 1),
                                  normalize=True)

                gen_image = generate_image(model=model, prior_distribution=distribution_fn, num_images=100,
                                           device=device)
                vutils.save_image(gen_image,
                                  '{}/gen_image_epoch_{}.png'.format(imagesdir, epoch + 1), normalize=True)


    plot_convergence(range(1, len(list_loss) + 1), list_loss, 'Test loss',
                     f'In testing loss convergence plot of {args.method}', imagesdir,
                     'test_loss_convergence.png')

    plot_convergence(range(1, len(train_list_loss) + 1), train_list_loss, 'Training loss',
                     f'In training loss convergence plot of {args.method}', imagesdir,
                     'train_loss_convergence.png')

    plot_convergence(range(1, len(list_RL) + 1), list_RL, 'Reconstruction Loss (RL)',
                     f'Reconstruction Loss (RL) convergence plot of {args.method}', imagesdir,
                     'rl_convergence.png')

    plot_convergence(range(1, len(list_WG) + 1), list_WG, 'Wasserstein Distance (WG)',
                     f'Wasserstein Distance (WG) convergence plot of {args.method}', imagesdir,
                     'wg_convergence.png')

    plot_convergence(range(1, len(list_LP) + 1), list_LP, 'Wasserstein Distance (LP)',
                     f'Wasserstein Distance (LP) convergence plot of {args.method}', imagesdir,
                     'lp_convergence.png')

    plot_convergence(range(1, len(list_F) + 1), list_F, 'Fairness (F)',
                     f'Fairness (F) convergence plot of {args.method}', imagesdir,
                     'f_convergence.png')

    plot_convergence(range(1, len(list_AD) + 1), list_AD, 'Averaging Distance (AD)',
                     f'Averaging Distance (AD) convergence plot of {args.method}', imagesdir,
                     'ad_convergence.png')

    plot_convergence(range(1, len(list_AD_images) + 1), list_AD_images, 'Averaging Distance in Images Space (ADI)',
                     f'Averaging Distance in Images Space (ADI) convergence plot of {args.method}', imagesdir,
                     'ad_images_convergence.png')

    plot_convergence(range(1, len(list_F_images) + 1), list_F_images, 'Fairness in Images Space (FI)',
                     f'Fairness in Images Space (FI) convergence plot of {args.method}', imagesdir,
                     'f_images_convergence.png')


if __name__ == '__main__':
    main()
