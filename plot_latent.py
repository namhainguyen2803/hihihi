import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)

from sklearn.manifold import TSNE

from swae.models.cifar10 import CIFAR10Autoencoder
from swae.models.mnist import MNISTAutoencoder
from swae.distributions import rand_cirlce2d, rand_ring2d, rand_uniform2d, rand, randn
import torch.optim as optim
import torchvision.utils as vutils
from dataloader.dataloader import *
from extract_result import extract_values
from utils import *
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')

    parser.add_argument('--batch-size', type=int, default=500, metavar='BS',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--batch-size-test', type=int, default=500, metavar='BST',
                        help='input batch size for evaluating (default: 500)')

    parser.add_argument('--weight_fsw', type=float, default=1,
                        help='weight of fsw (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--pretrained-model', type=str, metavar='S',
                        help='pretrained model path')
    parser.add_argument('--method', type=str, default='EFBSW', metavar='MED',
                        help='method (default: EFBSW)')
    parser.add_argument('--num-projections', type=int, default=10000, metavar='NP',
                        help='number of projections (default: 500)')

    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--lambda-obsw', type=float, default=1, metavar='OBSW',
                        help='hyper-parameter of OBSW method')
    parser.add_argument('--checkpoint-period', type=int, default=100, metavar='S',
                        help='checkpoint period (100, 200, 300)')
    
    args = parser.parse_args()
    
    if args.method == "OBSW" and args.lambda_obsw != 1:
        args.method = f"OBSW_{args.lambda_obsw}"
    
    checkpoint_periods = [args.checkpoint_period]
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    args.outdir = os.path.join(args.outdir, args.dataset)
    args.outdir = os.path.join(args.outdir, f"seed_{args.seed}")
    args.outdir = os.path.join(args.outdir, f"lr_{args.lr}")
    args.outdir = os.path.join(args.outdir, f"fsw_{args.weight_fsw}")
    args.outdir = os.path.join(args.outdir, args.method)
    outdir_checkpoint = os.path.join(args.outdir, "checkpoint")
    
    args.datadir = os.path.join(args.datadir, args.dataset)
    
    list_pretrained_models = list()
    print(f"Folder inside {outdir_checkpoint}: {os.listdir(outdir_checkpoint)}")
    if args.pretrained_model is None:
        for i in checkpoint_periods:
            pretrained_model_path = f"{outdir_checkpoint}/epoch_{i}/model/{args.dataset}_{args.method}.pth"
            check_path = os.path.isfile(pretrained_model_path)
            print(f"Check if pretrained model path {pretrained_model_path} exit or not: {check_path}")
            assert os.path.isfile(pretrained_model_path) == True, f"not exist {pretrained_model_path}"
            list_pretrained_models.append(pretrained_model_path)
    else:
        list_pretrained_models.append(args.pretrained_model)
    
    if args.dataset == "mnist":
        data_loader = MNISTLTDataLoader(data_dir=args.datadir, train_batch_size=args.batch_size, test_batch_size=args.batch_size_test)
        model = MNISTAutoencoder().to(device)
    elif args.dataset == "cifar10":
        data_loader = CIFAR10LTDataLoader(data_dir=args.datadir, train_batch_size=args.batch_size,
                                          test_batch_size=args.batch_size_test)
        model = CIFAR10Autoencoder(embedding_dim=args.embedding_size).to(device)
    else:
        raise ('dataset {} not supported'.format(args.dataset))
    train_loader, test_loader = data_loader.create_dataloader()
    
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
        
    METHOD_NAME = {
        "EFBSW": "es-MFSWB",
        "FBSW": "us-MFSWB",
        "lowerboundFBSW": "s-MFSWB",
        "OBSW_0.1": "MFSWB $\lambda = 0.1$",
        "OBSW": "MFSWB $\lambda = 1.0$",
        "OBSW_10.0": "MFSWB $\lambda = 10.0$",
        "BSW": "USWB",
        "None": "SWAE"
    }
    
    with torch.no_grad():
        for epoch in checkpoint_periods:
            
            pretrained_model_path = f"{outdir_checkpoint}/epoch_{i}/model/{args.dataset}_{args.method}.pth"
            print(model)
            model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
            
            test_encode, test_targets = list(), list()
            for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
                decoded_images, encoded_images = model(x_test.to(device))
                test_encode.append(encoded_images.detach())
                test_targets.append(y_test)

            test_encode, test_targets = torch.cat(test_encode), torch.cat(test_targets)
            test_encode, test_targets = test_encode.cpu().numpy(), test_targets.cpu().numpy()
            print(f"Shape of test dataset to plot: {test_encode.shape}, {test_targets.shape}")
            
            print('{}/test_latent.png'.format(f"{outdir_checkpoint}/epoch_{epoch}"))
            
            if args.dataset == "mnist":
                plt.figure(figsize=(10, 10))
                classes = np.unique(test_targets)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(classes)))
                for i, class_label in enumerate(classes):
                    plt.scatter(test_encode[test_targets == class_label, 0],
                                test_encode[test_targets == class_label, 1],
                                c=[colors[i]],
                                cmap=plt.cm.Spectral,
                                label=class_label)
                    
                plt.rc('text', usetex=True)
                plt.legend()
                title = f'{METHOD_NAME[args.method]}'
                plt.title(title, fontsize=20)
                plotted_dir = f"latentSpace/{args.dataset}/epoch_{args.checkpoint_period}/fsw_{args.weight_fsw}"
                os.makedirs(plotted_dir, exist_ok=True)
                plt.savefig('{}/epoch_{}_fsw_{}_method_{}.pdf'.format(plotted_dir, args.checkpoint_period, args.weight_fsw, args.method))

                # plt.axis("off")
                # plotted_path = f"plotted_images/{args.dataset}/seed_{args.seed}/fsw_{args.weight_fsw}/{args.method}/epoch_{args.checkpoint_period}"
                # os.makedirs(plotted_path, exist_ok=True)
                # plt.savefig('{}/epoch_{}_fsw_{}_method_{}.pdf'.format(plotted_path, args.checkpoint_period, args.weight_fsw, args.method))
                plt.close()
                
            elif args.dataset == "cifar10":
                tsne = TSNE(n_components=2, random_state=42)
                tsne_result = tsne.fit_transform(test_encode)
                classes = np.unique(test_targets)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(classes)))
                plt.figure(figsize=(10, 10))
                for i, class_label in enumerate(classes):
                    plt.scatter(tsne_result[test_targets == class_label, 0],
                                -tsne_result[test_targets == class_label, 1],
                                c=[colors[i]],
                                cmap=plt.cm.Spectral,
                                label=class_label)
                          
                plt.rc('text', usetex=True)
                plt.legend()
                title = f'{METHOD_NAME[args.method]}'
                plt.title(title, fontsize=20)
                plt.savefig('{}/epoch_{}_fsw_{}_method_{}.png'.format(f"{outdir_checkpoint}/epoch_{epoch}", args.checkpoint_period, args.weight_fsw, args.method))
                plt.close()

if __name__ == "__main__":
    main()