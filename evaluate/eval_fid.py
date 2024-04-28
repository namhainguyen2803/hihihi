import os
import shutil
import torch
from evaluate.eval_ws import compute_F_AD
from fid.fid_score import compute_fid_score, save_fid_stats
from utils import *
from metrics.wasserstein import *


def compute_WG(generated_images, real_images):
    return compute_fid_score(generated_images, real_images)


def compute_LP(encoded_samples, prior_samples):
    LP = compute_true_Wasserstein(X=encoded_samples[:10000], Y=prior_samples[:10000])
    return LP


def compute_RL(decoded_images, real_images):
    RL = torch.nn.functional.binary_cross_entropy(decoded_images, real_images)
    return RL


def compute_F_AD_images(gen_path, list_class_paths):
    list_distance = list()
    for cls_path in list_class_paths:
        dist = compute_fid_score(gen_path, cls_path)
        list_distance.append(dist)
    return compute_fairness(list_distance), compute_averaging_distance(list_distance)


def ultimate_evaluate_fid(args,
                          model,
                          test_loader,
                          prior_distribution,
                          device):
    with torch.no_grad():
        model.eval()

        list_labels = list()
        list_encoded_images = list()

        RL = 0
        total_images = 0

        for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
            list_labels.append(y_test)
            num_images = x_test.shape[0]
            total_images += num_images
            decoded_images, encoded_images = model(x_test.to(device))
            list_encoded_images.append(encoded_images.detach())
            RL += compute_RL(x_test.to(device), decoded_images.to(device)) * num_images

        tensor_labels = torch.cat(list_labels, dim=0).cpu()
        tensor_encoded_images = torch.cat(list_encoded_images, dim=0).cpu()
        tensor_generated_images = generate_image(model=model,
                                                 prior_distribution=prior_distribution,
                                                 num_images=total_images,
                                                 device=device).cpu()

        print(f"generated images have shape of: {tensor_generated_images.shape}")

        if len(os.listdir(args.gen_dir)) > 0:
            shutil.rmtree(args.gen_dir)
            os.makedirs(args.gen_dir, exist_ok=True)

        stat_gen_path = args.stat_gen_dir + ".npz"
        if os.path.isfile(stat_gen_path) == True:
            print(f"Found {stat_gen_path}, going to delete")
            os.remove(stat_gen_path)
            print(f"Found {stat_gen_path} or not: {os.path.isfile(stat_gen_path)}")

        make_jpg_images(tensor=tensor_generated_images, output_folder=args.gen_dir)
        paths = [args.gen_dir, stat_gen_path]
        save_fid_stats(paths=paths, batch_size=args.batch_size_test, device=device, dims=args.dims,
                       num_workers=args.num_workers)

        device = 'cpu'

        # Compute RL
        RL = RL / total_images
        print(f"RL: {RL}")

        # Compute WG
        WG = 0
        real_images_path = "stats/cifar10/ground_truth.npz"
        WG = compute_WG(stat_gen_path, real_images_path)
        print(f"WG: {WG}")

        # Compute LP
        LP = 0
        # prior_samples = prior_distribution(num_images)
        # LP = compute_LP(tensor_encoded_images, prior_samples)
        print(f"LP: {LP}")

        # Compute F and AD in latent space
        F, AD = compute_F_AD(list_features=tensor_encoded_images,
                             list_labels=tensor_labels,
                             prior_distribution=prior_distribution,
                             num_classes=args.num_classes,
                             device=device)
        print(f"F: {F}, AD: {AD}")

        # Compute F and AD in image space
        list_images_paths = list()
        for cls_id in range(args.num_classes):
            stat_cls = f"stats/cifar10/class_{cls_id}.npz"
            list_images_paths.append(stat_cls)
        F_images, AD_images = compute_F_AD_images(stat_gen_path, list_images_paths)
        print(f"FI: {F_images}, ADI: {AD_images}")

        RL = convert_to_cpu_number(RL)
        LP = convert_to_cpu_number(LP)
        WG = convert_to_cpu_number(WG)
        F = convert_to_cpu_number(F)
        AD = convert_to_cpu_number(AD)
        F_images = convert_to_cpu_number(F_images)
        AD_images = convert_to_cpu_number(AD_images)

        return RL, LP, WG, F, AD, F_images, AD_images
