import shutil

from metrics.wasserstein import *
import matplotlib.pyplot as plt
import os


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


def convert_to_cpu_number(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        return x.item()
    else:
        return x


def plot_convergence(iterations, data, ylabel, title, filename):
    plt.figure(figsize=(10, 10))
    plt.plot(iterations, data, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    return folder_path


def create_compression_file(images, sample_path):
    sample_path = recreate_folder(sample_path)
    if check_range_sigmoid(images):
        images = (images * 255).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        np.savez(sample_path, images)
        return sample_path
    else:
        raise ValueError('images must be in range [0, 1]')


def check_range_sigmoid(tensor):
    return (tensor >= 0).all() and (tensor <= 1).all()
