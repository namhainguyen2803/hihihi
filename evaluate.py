import os
from swae.utils import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from fid_evaluator import fid_evaluator_function

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

def fid_evaluation(model, prior_distribution, fid_stat, sample_path, device):
    # os.makedirs(sample_path)

    with torch.no_grad():
        gen_images = generate_image(model=model, prior_distribution=prior_distribution, num_images=50000, device=device)
        print(gen_images)
        print(check_range(gen_images))
        gen_images = (gen_images * 255).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        print(gen_images)
        np.savez(sample_path, gen_images)
        fid_evaluator_function(ref_batch=fid_stat, sample_batch=sample_path)

def check_range(tensor):
    return (tensor >= 0).all() and (tensor <= 1).all()