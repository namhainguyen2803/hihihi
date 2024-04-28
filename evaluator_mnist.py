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
from fid.inception import *
import torch.optim as optim
import torchvision.utils as vutils
from dataloader.dataloader import *
# from mnist_dataloader import *
from utils import *


def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')
    parser.add_argument('--images-path', default='/images/', help='path to images')

    parser.add_argument('--batch-size', type=int, default=500, metavar='BS',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--batch-size-test', type=int, default=500, metavar='BST',
                        help='input batch size for evaluating (default: 500)')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--weight_fsw', type=float, default=1,
                        help='weight of fsw (default: 1)')

    parser.add_argument('--stat-dir', default='stats', help='path to images')
    parser.add_argument('--method', type=str, default='EFBSW', metavar='MED',
                        help='method (default: EFBSW)')
    parser.add_argument('--num-projections', type=int, default=10000, metavar='NP',
                        help='number of projections (default: 500)')
    parser.add_argument('--embedding-size', type=int, default=48, metavar='ES',
                        help='embedding latent space (default: 48)')
    parser.add_argument("--dims", type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=("Dimensionality of Inception features to use. "
                              "By default, uses pool3 features"))
    
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
    
    parser.add_argument('--times', type=int, default=1, metavar='T',
                        help='...')
    
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
    
    if args.dataset != "mnist":
        
        args.images_path = os.path.join(args.images_path, args.dataset)
        args.images_path = os.path.join(args.images_path, f"seed_{args.seed}")
        args.gen_dir = os.path.join(args.images_path, f"lr_{args.lr}")
        args.gen_dir = os.path.join(args.gen_dir, f"fsw_{args.weight_fsw}")
        args.gen_dir = os.path.join(args.gen_dir, f"times_{args.times}")
        gen_dir = os.path.join(args.gen_dir, args.method)
        os.makedirs(gen_dir, exist_ok=True)
        args.gen_dir = gen_dir

        args.stat_dir = os.path.join(args.stat_dir, args.dataset)
        args.stat_dir = os.path.join(args.stat_dir, f"seed_{args.seed}")
        args.stat_gen_dir = os.path.join(args.stat_dir, f"lr_{args.lr}")
        args.stat_gen_dir = os.path.join(args.stat_gen_dir, f"fsw_{args.weight_fsw}")
        args.stat_gen_dir = os.path.join(args.stat_gen_dir, f"times_{args.times}")
        os.makedirs(args.stat_gen_dir, exist_ok=True)
        
        args.stat_gen_dir = f"{args.stat_gen_dir}/{args.method}"
    
    
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
        
    print(f"Folder inside {outdir_checkpoint}: {os.listdir(outdir_checkpoint)}")
    
    with torch.no_grad():
        for i in checkpoint_periods:
            
            pretrained_model_path = f"{outdir_checkpoint}/epoch_{i}/model/{args.dataset}_{args.method}.pth"
            check_path = os.path.isfile(pretrained_model_path)
            print(f"Check if pretrained model path {pretrained_model_path} exit or not: {check_path}")
            assert os.path.isfile(pretrained_model_path) == True, f"not exist {pretrained_model_path}"
            output_file = f'{args.outdir}/evaluate_epoch_{i}_{args.method}.log'
            print(model)
            if device == "cpu":
                model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(pretrained_model_path))
            RL, LP, WG, F, AD, F_images, AD_images = ultimate_evaluation(args=args,
                                                                        model=model,
                                                                        test_loader=test_loader,
                                                                        prior_distribution=distribution_fn,
                                                                        device=device)
            with open(output_file, 'a') as f:
                f.write(f"Evaluating pretrained model: {pretrained_model_path}:\n")
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