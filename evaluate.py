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
