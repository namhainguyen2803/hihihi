import os

import torch

from evaluate.eval_ws import compute_F_AD
from fid.fid_score import compute_fid_score
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
            RL += compute_RL(x_test, decoded_images) * num_images

        tensor_labels = torch.cat(list_labels, dim=0).cpu()
        tensor_encoded_images = torch.cat(list_encoded_images, dim=0).cpu()
        num_images = tensor_labels.shape[0]
        tensor_generated_images = generate_image(model=model,
                                                 prior_distribution=prior_distribution,
                                                 num_images=num_images,
                                                 device=device).cpu()

        num_jpg = 0
        for filename in os.listdir(args.gen_dir):
            if filename.lower().endswith('.jpg'):
                num_jpg += 1

        if num_jpg == total_images:
            print(f"Have already had generated image in {args.gen_dir}")
        elif num_jpg == 0:
            generated_images_path = make_jpg_images(tensor=tensor_generated_images, output_folder=args.gen_dir)
        else:
            raise Exception(f"Please delete {args.gen_dir} and run again")

        device = 'cpu'

        # Compute RL
        RL = RL / total_images
        print(f"RL: {RL}")

        # Compute WG
        real_images_path = f"{args.stat_dir}/ground_truth.npz"
        WG = compute_WG(generated_images_path, real_images_path)
        print(f"WG: {WG}")

        # Compute LP
        prior_samples = prior_distribution(num_images)
        print(tensor_encoded_images.shape, prior_samples.shape)
        LP = compute_LP(tensor_encoded_images, prior_samples)
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
            stat_cls = f"{args.stat_dir}/class_{cls_id}.npz"
            list_images_paths.append(stat_cls)
        F_images, AD_images = compute_F_AD_images(generated_images_path, list_images_paths)
        print(f"FI: {F_images}, ADI: {AD_images}")

        RL = convert_to_cpu_number(RL)
        LP = convert_to_cpu_number(LP)
        WG = convert_to_cpu_number(WG)
        F = convert_to_cpu_number(F)
        AD = convert_to_cpu_number(AD)
        F_images = convert_to_cpu_number(F_images)
        AD_images = convert_to_cpu_number(AD_images)

        return RL, LP, WG, F, AD, F_images, AD_images
