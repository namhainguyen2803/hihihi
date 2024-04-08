import os

import torch

from fid_evaluator import fid_evaluator_function
from swae.utils import *
import matplotlib.pyplot as plt

def generate_image(model,
                   prior_distribution,
                   num_images=100,
                   device='cpu'):
    with torch.no_grad():
        z_sample = prior_distribution(num_images).to(device)
        model_device = next(model.parameters()).device
        x_synthesis = model.generate(z_sample.to(model_device)).to(device)
        return x_synthesis


def compute_fairness(list_metric):
    num_classes = len(list_metric)
    dist_pairwise = list()
    for cls_id_i in range(num_classes):
        for cls_id_j in range(cls_id_i + 1, num_classes):
            a = abs(list_metric[cls_id_i] - list_metric[cls_id_j])
            dist_pairwise.append(a)
    return torch.sum(torch.tensor(dist_pairwise)) / len(dist_pairwise)


def compute_averaging_distance(list_metric):
    return torch.sum(torch.tensor(list_metric)) / len(list_metric)


def compute_fairness_and_averaging_distance(list_features,
                                            list_labels,
                                            prior_distribution,
                                            num_classes,
                                            device):
    with torch.no_grad():
        dist_swd = list()
        for cls_id in range(num_classes):
            features_cls = list_features[list_labels == cls_id]
            z_samples = prior_distribution(features_cls.shape[0]).to(device)
            wd = compute_true_Wasserstein(X=features_cls, Y=z_samples)

            SAMPLE_PATH = 'generated_images/cifar10_epoch_{}.npz'.format(epoch + 1)
            fid_evaluation(model=model, prior_distribution=distribution_fn,
                           fid_stat="/kaggle/input/cifar10/cifar10.npz", sample_path=SAMPLE_PATH, device=device)

            dist_swd.append(wd)
    return compute_fairness(dist_swd), compute_averaging_distance(dist_swd)


def compute_fairness_and_averaging_distance_in_images_space(model,
                                                            prior_distribution,
                                                            tensor_flatten_real_images,
                                                            tensor_labels,
                                                            num_classes,
                                                            device):
    each_class_images = dict()
    for cls_id in range(num_classes):
        if cls_id in each_class_images:
            each_class_images[cls_id].append(tensor_flatten_real_images[tensor_labels == cls_id])
        else:
            each_class_images[cls_id] = list()
            each_class_images[cls_id].append(tensor_flatten_real_images[tensor_labels == cls_id])

    list_ws_distance = list()
    for cls_id, list_flatten_real_images in each_class_images.items():
        flatten_real_images = torch.cat(list_flatten_real_images, dim=0).cpu()
        num_images = len(flatten_real_images)
        generated_images = generate_image(model=model,
                                          prior_distribution=prior_distribution,
                                          num_images=num_images,
                                          device=device)
        flatten_generated_images = generated_images.reshape(num_images, -1)
        ws = compute_true_Wasserstein(X=flatten_generated_images, Y=flatten_real_images)
        list_ws_distance.append(ws)

    return compute_fairness(list_ws_distance), compute_averaging_distance(list_ws_distance)


def ultimate_evaluation_fid(args,
                            model,
                            evaluator,
                            test_loader,
                            prior_distribution,
                            device='cpu'):
    pass

def ultimate_evaluation(args,
                        model,
                        evaluator,
                        test_loader,
                        prior_distribution,
                        device='cpu'):
    with torch.no_grad():
        model.eval()

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
                                                 prior_distribution=prior_distribution,
                                                 num_images=tensor_real_images.shape[0],
                                                 device=device).cpu()

        num_images = tensor_real_images.shape[0]

        tensor_flatten_real_images = tensor_real_images.view(num_images, -1)
        tensor_flatten_generated_images = tensor_generated_images.view(num_images, -1)

        device = 'cpu'

        # Compute RL
        RL = torch.nn.functional.binary_cross_entropy(tensor_decoded_images, tensor_real_images)

        # Compute WG
        WG = compute_true_Wasserstein(X=tensor_flatten_generated_images, Y=tensor_flatten_real_images)

        # Compute LP
        prior_samples = prior_distribution(num_images).to(device)
        LP = compute_true_Wasserstein(X=tensor_encoded_images, Y=prior_samples)

        # Compute F and AD in latent space
        F, AD = compute_fairness_and_averaging_distance(list_features=tensor_encoded_images,
                                                        list_labels=tensor_labels,
                                                        prior_distribution=prior_distribution,
                                                        num_classes=args.num_classes,
                                                        device=device)

        # Compute F and AD in image space
        F_images, AD_images = compute_fairness_and_averaging_distance_in_images_space(model=model,
                                                                                      prior_distribution=prior_distribution,
                                                                                      tensor_flatten_real_images=tensor_flatten_real_images,
                                                                                      tensor_labels=tensor_labels,
                                                                                      num_classes=args.num_classes,
                                                                                      device=device)

        RL = convert_to_cpu_number(RL)
        LP = convert_to_cpu_number(LP)
        WG = convert_to_cpu_number(WG)
        F = convert_to_cpu_number(F)
        AD = convert_to_cpu_number(AD)
        F_images = convert_to_cpu_number(F_images)
        AD_images = convert_to_cpu_number(AD_images)

        return RL, LP, WG, F, AD, F_images, AD_images


def convert_to_cpu_number(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        return x.item()
    else:
        return x


def plot_convergence(iterations, data, ylabel, title, imagesdir, filename):
    plt.figure(figsize=(10, 10))
    plt.plot(iterations, data, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'{imagesdir}/{filename}')
    plt.close()


def fid_evaluation(model, prior_distribution, fid_stat, sample_path, device):
    os.makedirs(sample_path)
    with torch.no_grad():
        gen_images = generate_image(model=model, prior_distribution=prior_distribution, num_images=50000, device=device)
        print(check_range(gen_images))
        gen_images = (gen_images * 255).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        np.savez(sample_path, gen_images)
        fid_evaluator_function(ref_batch=fid_stat, sample_batch=sample_path)

def check_range(tensor):
    return (tensor >= 0).all() and (tensor <= 1).all()
