import argparse
import matplotlib as mpl

mpl.use('Agg')
from sklearn.manifold import TSNE

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
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    parser.add_argument('--weight_swd', type=float, default=1,
                        help='weight of swd (default: 1)')
    parser.add_argument('--weight_fsw', type=float, default=1,
                        help='weight of fsw (default: 1)')
    parser.add_argument('--method', type=str, default='FEFBSW', metavar='MED',
                        help='method (default: FEFBSW)')
    parser.add_argument('--num-projections', type=int, default=10000, metavar='NP',
                        help='number of projections (default: 500)')
    parser.add_argument('--embedding-size', type=int, default=48, metavar='ES',
                        help='embedding latent space (default: 48)')

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

    args.outdir = os.path.join(args.outdir, f"lr_{args.lr}")
    args.outdir = os.path.join(args.outdir, f"fsw_{args.weight_fsw}")
    args.outdir = os.path.join(args.outdir, args.method)

    outdir_best = os.path.join(args.outdir, "best")
    outdir_end = os.path.join(args.outdir, "end")
    outdir_convergence = os.path.join(args.outdir, "convergence")

    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(outdir_best, exist_ok=True)
    os.makedirs(outdir_end, exist_ok=True)
    os.makedirs(outdir_convergence, exist_ok=True)
    os.makedirs("statistic", exist_ok=True)
    # determine device and device dep. args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    output_file = f'{args.outdir}/output.log'

    # log args
    with open(output_file, 'a') as f:
        # log args
        if args.optimizer == 'rmsprop':
            f.write(
                'batch size {}\nepochs {}\nRMSprop lr {} alpha {}\ndistribution {}\nusing device {}\nseed set to {}\n'.format(
                    args.batch_size, args.epochs, args.lr, args.alpha, args.distribution, device.type, args.seed
                ))
        else:
            f.write(
                'batch size {}\nepochs {}\n{}: lr {} betas {}/{}\ndistribution {}\nusing device {}\nseed set to {}\n'.format(
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
        if args.dataset == 'mnist':
            RL, LP, WG, F, AD, F_images, AD_images = ultimate_evaluation(args=args,
                                                                         model=model,
                                                                         test_loader=test_loader,
                                                                         prior_distribution=distribution_fn,
                                                                         device=device)
        else:
            RL, LP, WG, F, AD, F_images, AD_images = ultimate_evaluate_fid(args=args,
                                                                           model=model,
                                                                           test_loader=test_loader,
                                                                           prior_distribution=distribution_fn,
                                                                           device=device)
        with open(output_file, 'a') as f:
            f.write("In pre-training, when evaluating test loader:\n")
            f.write(f" +) Reconstruction loss (RL): {RL}\n")
            f.write(f" +) Wasserstein distance between generated and real images (WG): {WG}\n")
            f.write(f" +) Wasserstein distance between posterior and prior distribution (LP): {LP}\n")
            f.write(f" +) Fairness (F): {F}\n")
            f.write(f" +) Averaging distance (AD): {AD}\n")
            f.write(f" +) Fairness in images space (FI): {F_images}\n")
            f.write(f" +) Averaging distance in images space (ADI): {AD_images}\n")
            f.write("\n")

        list_RL.append(RL)
        list_WG.append(WG)
        list_LP.append(LP)
        list_F.append(F)
        list_AD.append(AD)
        list_AD_images.append(AD_images)
        list_F_images.append(F_images)

    eval_best = 1000

    print()
    # train networks for n epochs
    for epoch in range(args.epochs):
        with open(output_file, 'a') as f:
            f.write('training...\n')
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
                with open(output_file, 'a') as f:
                    f.write('Train Epoch: {} ({:.2f}%) [{}/{}]\tLoss: {:.6f}\n'.format(
                        epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                        (batch_idx + 1), len(train_loader),
                        batch['loss'].item()))

        with open(output_file, 'a') as f:
            f.write('evaluating...\n')
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

        if (epoch + 1) % args.log_epoch_interval == 0 or (epoch + 1) == args.epochs:

            if args.dataset == 'mnist':
                RL, LP, WG, F, AD, F_images, AD_images = ultimate_evaluation(args=args,
                                                                             model=model,
                                                                             test_loader=test_loader,
                                                                             prior_distribution=distribution_fn,
                                                                             device=device)
            else:
                RL, LP, WG, F, AD, F_images, AD_images = ultimate_evaluate_fid(args=args,
                                                                               model=model,
                                                                               test_loader=test_loader,
                                                                               prior_distribution=distribution_fn,
                                                                               device=device)

            with open(output_file, 'a') as f:
                f.write('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}\n'.format(epoch + 1,
                                                                          float(epoch + 1) / (args.epochs) * 100.,
                                                                          test_loss))
                f.write('{{"metric": "loss", "value": {}}}\n'.format(test_loss))
                f.write("When evaluating test loader:\n")
                f.write(f" +) Reconstruction loss (RL): {RL}\n")
                f.write(f" +) Wasserstein distance between generated and real images (WG): {WG}\n")
                f.write(f" +) Wasserstein distance between posterior and prior distribution (LP): {LP}\n")
                f.write(f" +) Fairness (F): {F}\n")
                f.write(f" +) Averaging distance (AD): {AD}\n")
                f.write(f" +) Fairness in images space (FI): {F_images}\n")
                f.write(f" +) Averaging distance in images space (ADI): {AD_images}\n")
                f.write("\n")

            list_RL.append(RL)
            list_WG.append(WG)
            list_LP.append(LP)
            list_F.append(F)
            list_AD.append(AD)
            list_AD_images.append(AD_images)
            list_F_images.append(F_images)

            # update best or end
            if (epoch + 1) == args.epochs or eval_best > F + AD:

                if (epoch + 1) == args.epochs:
                    imagesdir_epoch = os.path.join(outdir_end, "images")
                    chkptdir_epoch = os.path.join(outdir_end, "model")
                    with open(output_file, 'a') as f:
                        f.write(
                            f"Saving end model in final epoch {epoch}, the result: F = {F}, W = {AD}, F_images = {F_images}, W_images = {AD_images}\n")
                else:
                    imagesdir_epoch = os.path.join(outdir_best, "images")
                    chkptdir_epoch = os.path.join(outdir_best, "model")
                    eval_best = F + AD
                    with open(output_file, 'a') as f:
                        f.write(
                            f"Saving best model in epoch {epoch}, the result: F = {F}, W = {AD}, F_images = {F_images}, W_images = {AD_images}\n")

                os.makedirs(imagesdir_epoch, exist_ok=True)
                os.makedirs(chkptdir_epoch, exist_ok=True)

                torch.save(model.state_dict(), '{}/{}.pth'.format(chkptdir_epoch, args.dataset))
                test_encode, test_targets = torch.cat(test_encode), torch.cat(test_targets)
                test_encode, test_targets = test_encode.cpu().numpy(), test_targets.cpu().numpy()

                if args.dataset == "mnist":
                    # plot
                    plt.figure(figsize=(10, 10))
                    plt.scatter(test_encode[:, 0], -test_encode[:, 1], c=(10 * test_targets), cmap=plt.cm.Spectral)
                    plt.xlim([-1.5, 1.5])
                    plt.ylim([-1.5, 1.5])
                    title = f'Latent Space of {args.method} method'
                    plt.title(title)
                    plt.savefig('{}/test_latent.png'.format(imagesdir_epoch))
                    plt.close()

                else:
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_result = tsne.fit_transform(test_encode)

                    plt.figure(figsize=(10, 10))
                    plt.scatter(tsne_result[:, 0], -tsne_result[:, 1], c=(10 * test_targets), cmap=plt.cm.Spectral)
                    plt.xlim([-1.5, 1.5])
                    plt.ylim([-1.5, 1.5])
                    title = f'Latent Space of {args.method} method'
                    plt.title(title)
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.savefig('{}/test_latent.png'.format(imagesdir_epoch))
                    plt.colorbar(label='Target')
                    plt.close()

                # save sample input and reconstruction
                vutils.save_image(x_test,
                                  '{}/{}_test_samples.png'.format(imagesdir_epoch, args.distribution))

                vutils.save_image(test_evals['decode'].detach(),
                                  '{}/{}_test_recon.png'.format(imagesdir_epoch, args.distribution),
                                  normalize=True)

                vutils.save_image(x, '{}/{}_train_samples.png'.format(imagesdir_epoch, args.distribution))

                vutils.save_image(batch['decode'].detach(),
                                  '{}/{}_train_recon.png'.format(imagesdir_epoch, args.distribution),
                                  normalize=True)

                gen_image = generate_image(model=model, prior_distribution=distribution_fn, num_images=100,
                                           device=device)
                vutils.save_image(gen_image,
                                  '{}/gen_image.png'.format(imagesdir_epoch), normalize=True)

    plot_convergence(range(1, len(list_loss) + 1), list_loss, 'Test loss',
                     f'In testing loss convergence plot of {args.method}',
                     f"{outdir_convergence}/test_loss_convergence.png")

    plot_convergence(range(1, len(train_list_loss) + 1), train_list_loss, 'Training loss',
                     f'In training loss convergence plot of {args.method}',
                     f"{outdir_convergence}/train_loss_convergence.png")

    plot_convergence(range(1, len(list_RL) + 1), list_RL, 'Reconstruction Loss (RL)',
                     f'Reconstruction Loss (RL) convergence plot of {args.method}',
                     f"{outdir_convergence}/rl_convergence.png")

    plot_convergence(range(1, len(list_WG) + 1), list_WG, 'Wasserstein Distance (WG)',
                     f'Wasserstein Distance (WG) convergence plot of {args.method}',
                     f"{outdir_convergence}/wg_convergence.png")

    plot_convergence(range(1, len(list_LP) + 1), list_LP, 'Wasserstein Distance (LP)',
                     f'Wasserstein Distance (LP) convergence plot of {args.method}',
                     f"{outdir_convergence}/lp_convergence.png")

    plot_convergence(range(1, len(list_F) + 1), list_F, 'Fairness (F)',
                     f'Fairness (F) convergence plot of {args.method}',
                     f"{outdir_convergence}/f_convergence.png")

    # Modify the last call to have the desired pattern for output file path
    plot_convergence(range(1, len(list_loss) + 1), list_loss, 'Test loss',
                     f'In testing loss convergence plot of {args.method}',
                     f"{outdir_convergence}/test_loss_convergence.png")


if __name__ == '__main__':
    main()
