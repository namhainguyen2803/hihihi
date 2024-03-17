import argparse
import os

import matplotlib as mpl

from evaluate import wasserstein_evaluation

mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.utils as vutils
from swae.distributions import rand_cirlce2d, rand_ring2d, rand_uniform2d
from swae.models.mnist import MNISTAutoencoder
from swae.trainer import SWAEBatchTrainer
from torchvision import datasets, transforms
from swae.dataloader import *
from swae.utils import *

def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch MNIST Example')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')

    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='A',
                        help='RMSprop alpha/rho (default: 0.9)')
    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                        help='Optimizer (default: rmsprop)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches to log training status (default: 10)')
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
    print('batch size {}\nepochs {}\nRMSprop lr {} alpha {}\ndistribution {}\nusing device {}\nseed set to {}'.format(
        args.batch_size, args.epochs, args.lr, args.alpha, args.distribution, device.type, args.seed
    ))

    # build train and test set data loaders
    data_loader = MNISTDataLoader(train_batch_size=args.batch_size, test_batch_size=args.batch_size)
    train_loader, test_loader = data_loader.create_dataloader()

    # create encoder and decoder
    model = MNISTAutoencoder().to(device)
    print(model)
    # create optimizer
    if args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # determine latent distribution
    if args.distribution == 'circle':
        distribution_fn = rand_cirlce2d
    elif args.distribution == 'ring':
        distribution_fn = rand_ring2d
    else:
        distribution_fn = rand_uniform2d

    # create batch sliced_wasserstein autoencoder trainer
    trainer = SWAEBatchTrainer(autoencoder=model, optimizer=optimizer, num_classes=10, distribution_fn=distribution_fn,
                               device=device)
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

        # evalutate_fid autoencoder on test dataset
        # keep track swd gap of each gap, reconstruction loss of each gap
        test_encode, test_targets, test_loss = list(), list(), 0.0
        posterior_gap = [0 for _ in range(data_loader.num_classes)]
        reconstruction_loss = [0 for _ in range(data_loader.num_classes)]
        list_l1 = [0 for _ in range(data_loader.num_classes)]
        num_instances = [0 for _ in range(data_loader.num_classes)]

        with torch.no_grad():
            for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
                test_evals = trainer.test_on_batch(x_test, y_test)

                test_encode.append(test_evals['encode'].detach())
                test_loss += test_evals['loss'].item()
                test_targets.append(y_test)

                # update evaluation incrementally
                for cls_id in range(data_loader.num_classes):
                    if cls_id in test_evals['list_recon'].keys():
                        reconstruction_loss[cls_id] += test_evals['list_recon'][cls_id]

                    if cls_id in test_evals['list_swd'].keys():
                        posterior_gap[cls_id] += test_evals['list_swd'][cls_id]

                    if cls_id in test_evals['list_l1'].keys():
                        list_l1[cls_id] += test_evals['list_l1'][cls_id]

                    num_instances[cls_id] += x_test[y_test == cls_id].shape[0]

            ws_score = wasserstein_evaluation(model=model, prior_distribution=distribution_fn, test_loader=test_loader,
                                              device=device)

            print()
            print("############## EVALUATION ##############")
            print("Overall evaluation results:")
            print(f"Overall loss: {test_evals['loss'].item()}")
            print(f"Wasserstein distance between generated images and real images: {ws_score}")
            print(f"Reconstruction loss: {test_evals['recon_loss'].item()}")
            print(f"SWD loss: {test_evals['swd_loss'].item()}")
            print(f"Fair_SWD loss: {test_evals['fsw_loss'].item()}")
            print(f"L1 loss: {test_evals['l1_loss'].item()}")

            print()
            print("Evaluation of each class:")
            print(f"Reconstruction loss: {reconstruction_loss}")
            print(f"L1 loss: {list_l1}")
            print(f"Posterior gap: {posterior_gap}")

            print("########################################")
            print()

        # plot
        test_encode, test_targets = torch.cat(test_encode).cpu().numpy(), torch.cat(test_targets).cpu().numpy()

        test_loss /= len(test_loader)
        print('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}'.format(
            epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
            test_loss))
        print('{{"metric": "loss", "value": {}}}'.format(test_loss))
        # save model
        torch.save(model.state_dict(), '{}/mnist_epoch_{}.pth'.format(chkptdir, epoch + 1))
        # save encoded samples plot
        plt.figure(figsize=(10, 10))
        plt.scatter(test_encode[:, 0], -test_encode[:, 1], c=(10 * test_targets), cmap=plt.cm.Spectral)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Test Latent Space\nLoss: {:.5f}'.format(test_loss))
        plt.savefig('{}/test_latent_epoch_{}.png'.format(imagesdir, epoch + 1))
        plt.close()
        # save sample input and reconstruction
        vutils.save_image(x, '{}/test_samples_epoch_{}.png'.format(imagesdir, epoch + 1))
        vutils.save_image(batch['decode'].detach(),
                          '{}/test_recon_epoch_{}.png'.format(imagesdir, epoch + 1),
                          normalize=True)

        gen_image = generate_image(model=model, prior_distribution=distribution_fn, num_images=30, device=device)
        vutils.save_image(gen_image,
                          '{}/gen_image_epoch_{}.png'.format(imagesdir, epoch + 1), normalize=True)
if __name__ == '__main__':
    main()
