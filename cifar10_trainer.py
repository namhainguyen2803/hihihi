import argparse
import os

import torch
import torch.optim as optim
import torchvision.utils as vutils
from swae.distributions import rand, randn
from swae.models.cifar10 import CIFAR10Autoencoder
from swae.trainer import SWAEBatchTrainer
from torchvision import datasets, transforms
from swae.dataloader import *
from swae.utils import *

def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch CIFAR10 Example')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='A',
                        help='RMSprop alpha/rho (default: 0.9)')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='B1',
                        help='Adam beta1 (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam beta2 (default: 0.999)')
    parser.add_argument('--num-projections', type=int, default=500, metavar='NP',
                        help='number of projections (default: 500)')
    parser.add_argument('--weight', type=float, default=10.0, metavar='W',
                        help='weight of divergence (default: 10.0)')
    parser.add_argument('--distribution', type=str, default='normal', metavar='DIST',
                        help='Latent Distribution (default: normal)')
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                        help='Optimizer (default: rmsprop)')
    parser.add_argument('--embedding-size', type=int, default=64, metavar='ES',
                        help='model embedding size (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=16, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 16)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                        help='number of batches to log training status (default: 25)')
    parser.add_argument('--log-epoch-interval', type=int, default=1, metavar='N',
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
    dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {'num_workers': args.num_workers, 'pin_memory': False}
    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    # log args
    print('batch size {}\nepochs {}\nAdam: lr {} betas {}/{}\ndistribution {}\nusing device {}\nseed set to {}'.format(
        args.batch_size, args.epochs,
        args.lr, args.beta1, args.beta2, args.distribution,
        device.type, args.seed
    ))


    # build train and test set data loaders
    data_loader = CIFAR10DataLoader(train_batch_size=args.batch_size, test_batch_size=args.batch_size, dataloader_kwargs=dataloader_kwargs)
    train_loader, test_loader = data_loader.create_dataloader()

    # create encoder and decoder
    model = CIFAR10Autoencoder(embedding_dim=args.embedding_size).to(device)
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
    if args.distribution == 'uniform':
        distribution_fn = rand(args.embedding_size)
    elif args.distribution == 'normal':
        distribution_fn = randn(args.embedding_size)
    else:
        raise('distribution {} not supported'.format(args.distribution))

    # create batch sliced_wasserstein autoencoder trainer
    trainer = SWAEBatchTrainer(autoencoder=model, optimizer=optimizer, distribution_fn=distribution_fn, num_classes=10,
                               num_projections=args.num_projections, weight=args.weight, device=device)

    # put networks in training mode
    model.train()
    # train networks for n epochs
    print('training...')
    for epoch in range(args.epochs):
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

        # evaluate autoencoder on test dataset
        test_encode, test_targets, test_loss = list(), list(), 0.0
        with torch.no_grad():
            for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
                test_evals = trainer.test_on_batch(x_test, y_test)
                test_encode.append(test_evals['encode'].detach())
                test_loss += test_evals['loss'].item()
                test_targets.append(y_test)

        test_loss /= len(test_loader)
        print('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}'.format(
                epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                test_loss))
        print('{{"metric": "loss", "value": {}}}'.format(test_loss))
        # save artifacts ever log epoch interval
        if (epoch + 1) % args.log_epoch_interval == 0:
            # save model
            torch.save(model.state_dict(), '{}/cifar10_epoch_{}.pth'.format(chkptdir, epoch + 1))
            # save sample input and reconstruction
            vutils.save_image(x, '{}/{}_test_samples_epoch_{}.png'.format(imagesdir, args.distribution, epoch + 1))
            vutils.save_image(batch['decode'].detach(),
                              '{}/{}_test_recon_epoch_{}.png'.format(imagesdir, args.distribution, epoch + 1),
                              normalize=True)


if __name__ == '__main__':
    main()
