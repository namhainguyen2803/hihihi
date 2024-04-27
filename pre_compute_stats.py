import argparse
# from evaluate.eval_fid import *
from dataloader.dataloader import CIFAR10LTDataLoader
from utils import make_jpg_images
from fid.fid_score import save_fid_stats
from fid.inception import *
import os


def compute_statistics(args, device):
    if args.dataset == 'cifar10':
        data_loader = CIFAR10LTDataLoader(data_dir=args.datadir, train_batch_size=args.batch_size,
                                          test_batch_size=args.batch_size_test)
    else:
        data_loader = None
    train_loader, test_loader = data_loader.create_dataloader()

    list_real_images = list()
    list_class_jpg = dict()
    for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
        list_real_images.append(x_test)
        for cls_id in range(10):
            class_jpg = make_jpg_images(tensor=x_test[y_test == cls_id],
                                        output_folder=f"{args.images_path}/class_{cls_id}")
            list_class_jpg[cls_id] = class_jpg

    tensor_real_images = torch.cat(list_real_images, dim=0).cpu()
    real_images_path = make_jpg_images(tensor=tensor_real_images, output_folder=f"{args.images_path}/ground_truth")

    dataset_path = os.path.join(args.stat_dir, args.dataset)
    os.makedirs(dataset_path, exist_ok=True)

    for i in range(args.num_classes):
        paths = [list_class_jpg[i], f"{args.stat_dir}/{args.dataset}/class_{i}"]
        save_fid_stats(paths=paths, batch_size=args.batch_size_test, device=device, dims=args.dims,
                       num_workers=args.num_workers)

    paths = [f"{args.images_path}/ground_truth", f"{args.stat_dir}/ground_truth"]
    save_fid_stats(paths=paths, batch_size=args.batch_size_test, device=device, dims=args.dims,
                   num_workers=args.num_workers)

    return list_class_jpg, real_images_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch')
    parser.add_argument('--dataset', default='cifar10', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--images-path', default='images', help='path to images')
    
    parser.add_argument('--datadir', default='data', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')

    parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--batch-size-test', type=int, default=128, metavar='BST',
                        help='input batch size for evaluating (default: 500)')

    parser.add_argument("--dims", type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=("Dimensionality of Inception features to use. "
                              "By default, uses pool3 features"))

    parser.add_argument('--stat-dir', default='stats', help='path to images')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    args.datadir = os.path.join(args.datadir, args.dataset)

    compute_statistics(args=args, device=device)
