import torch
from swae.utils import *


def generate_image(model, prior_distribution, num_images, device='cpu'):
    with torch.no_grad():
        z_sample = prior_distribution(num_images).to(device)
        x_synthesis = model.generate(z_sample).to(device)
        return x_synthesis


def wasserstein_evaluation(model, prior_distribution, test_loader, device):
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
        ws = compute_true_Wasserstein(X=list_generated_images, Y=list_real_images, p=2)

        return ws


def calculate_fairness(list_metric, p=2):
    tensor_ = torch.tensor(list_metric)
    avg_ = torch.sum(tensor_) / len(list_metric)
    return torch.sum(torch.pow(torch.abs((tensor_ - avg_)), p))


def calculate_pairwise_swd(list_features, list_labels, num_classes, device, num_projections=200):
    with torch.no_grad():
        dist_pairwise = list()
        features_dict = dict()
        for cls_id in range(num_classes):
            features_dict[cls_id] = list_features[list_labels == cls_id]

        for cls_id_i in range(num_classes):
            for cls_id_j in range(cls_id_i + 1, num_classes):
                swd = sliced_wasserstein_distance(encoded_samples=features_dict[cls_id_i],
                                                  distribution_samples=features_dict[cls_id_j],
                                                  num_projections=200,
                                                  p=2,
                                                  device=device)
                dist_pairwise.append(swd)

    return torch.sum(torch.tensor(dist_pairwise)) / len(dist_pairwise)


def calculate_pairwise_swd_2(list_features, list_labels, prior_distribution, num_classes, device, num_projections=200):
    with torch.no_grad():
        dist_swd = dict()
        for cls_id in range(num_classes):
            features_cls = list_features[list_labels == cls_id]
            z_samples = prior_distribution(features_cls.shape[0]).to(device)

            swd = sliced_wasserstein_distance(encoded_samples=features_cls,
                                              distribution_samples=z_samples,
                                              num_projections=num_projections,
                                              p=2,
                                              device=device)

            dist_swd[cls_id] = swd

        dist_pairwise = list()
        for cls_id_i in range(num_classes):
            for cls_id_j in range(cls_id_i + 1, num_classes):
                a = torch.abs(dist_swd[cls_id_i] - dist_swd[cls_id_j])
                dist_pairwise.append(a)

    return torch.sum(torch.tensor(dist_pairwise)) / len(dist_pairwise)
