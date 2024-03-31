import torch
from swae.utils import *


def generate_image(model,
                   prior_distribution,
                   num_images=100,
                   device='cpu'):
    with torch.no_grad():
        z_sample = prior_distribution(num_images).to(device)
        x_synthesis = model.generate(z_sample).to(device)
        return x_synthesis


def wasserstein_distance_of_generated_images_and_ground_truth_images(model,
                                                                     prior_distribution,
                                                                     test_loader,
                                                                     device,
                                                                     theta,
                                                                     num_projections=10000):
    with torch.no_grad():
        list_real_images = list()
        list_generated_images = list()
        model.eval()
        for images, labels in test_loader:
            images = images.to(device)

            num_images = images.shape[0]
            generated_images = generate_image(model=model, prior_distribution=prior_distribution, num_images=num_images,
                                              device=device)

            flatten_images = images.reshape(num_images, -1)
            flatten_generated_images = generated_images.reshape(num_images, -1)

            list_real_images.append(flatten_images)
            list_generated_images.append(flatten_generated_images)

        list_real_images = torch.cat(list_real_images, dim=0).cpu()
        list_generated_images = torch.cat(list_generated_images, dim=0).cpu()

        ws = sliced_wasserstein_distance(encoded_samples=list_generated_images,
                                         distribution_samples=list_real_images,
                                         num_projections=num_projections,
                                         p=2,
                                         device='cpu',
                                         theta=theta)

        return ws


def compute_fairness(list_metric):
    num_classes = len(list_metric)
    dist_pairwise = list()
    for cls_id_i in range(num_classes):
        for cls_id_j in range(cls_id_i + 1, num_classes):
            a = torch.abs(list_metric[cls_id_i] - list_metric[cls_id_j])
            dist_pairwise.append(a)
    return torch.sum(torch.tensor(dist_pairwise)) / len(dist_pairwise)


def compute_averaging_distance(list_metric):
    return torch.sum(torch.tensor(list_metric)) / len(list_metric)


def compute_fairness_and_averaging_distance(list_features,
                                            list_labels,
                                            prior_distribution,
                                            num_classes,
                                            device,
                                            num_projections=10000,
                                            dim=2,
                                            theta=None):
    with torch.no_grad():
        dist_swd = list()
        if theta is None:
            theta = rand_projections(dim=dim, num_projections=10000, device='cpu')
        for cls_id in range(num_classes):
            features_cls = list_features[list_labels == cls_id]
            z_samples = prior_distribution(features_cls.shape[0]).to(device)
            swd = sliced_wasserstein_distance(encoded_samples=features_cls,
                                              distribution_samples=z_samples,
                                              num_projections=num_projections,
                                              p=2,
                                              device=device,
                                              theta=theta)
            dist_swd.append(swd)
    return compute_fairness(dist_swd), compute_averaging_distance(dist_swd)


def wasserstein_distance_of_generated_images_and_each_class_ground_truth_images(model,
                                                                                prior_distribution,
                                                                                test_loader,
                                                                                device,
                                                                                num_projections=10000,
                                                                                theta=None):
    model.eval()
    with torch.no_grad():
        list_real_images = dict()
        list_ws_distance = list()

        images_dim = 0
        for images, labels in test_loader:
            images = images.to(device)
            num_images = images.shape[0]
            images_dim = torch.prod(images.shape[1:])
            flatten_images = images.reshape(num_images, -1)

            for cls_id in range(torch.unique(labels)):
                if cls_id in list_real_images:
                    list_real_images[cls_id].append(flatten_images[labels == cls_id])
                else:
                    list_real_images[cls_id] = list()
                    list_real_images[cls_id].append(flatten_images[labels == cls_id])

        if theta is None:
            theta = rand_projections(dim=images_dim, num_projections=10000, device='cpu')

        for cls_id, list_flatten_real_images in list_real_images.items():
            flatten_real_images = torch.cat(list_flatten_real_images, dim=0).cpu()
            num_images = len(flatten_real_images)
            generated_images = generate_image(model=model,
                                              prior_distribution=prior_distribution,
                                              num_images=num_images,
                                              device=device)
            flatten_generated_images = generated_images.reshape(num_images, -1)

            ws = sliced_wasserstein_distance(encoded_samples=flatten_real_images,
                                             distribution_samples=flatten_generated_images,
                                             num_projections=num_projections,
                                             p=2,
                                             device='cpu',
                                             theta=theta)

            list_ws_distance.append(ws)

        return list_ws_distance


def compute_fairness_and_averaging_distance_in_images_space(model,
                                                            prior_distribution,
                                                            test_loader,
                                                            device,
                                                            num_projections=10000,
                                                            theta=None):
    list_ws_distance = wasserstein_distance_of_generated_images_and_each_class_ground_truth_images(model=model,
                                                                                                   prior_distribution=prior_distribution,
                                                                                                   test_loader=test_loader,
                                                                                                   device=device,
                                                                                                   num_projections=num_projections,
                                                                                                   theta=theta)
    return compute_fairness(list_ws_distance), compute_averaging_distance(list_ws_distance)