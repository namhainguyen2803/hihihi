from metrics.wasserstein import *
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import zipfile


def generate_image(model,
                   prior_distribution,
                   num_images=100,
                   device='cpu'):
    with torch.no_grad():
        z_sample = prior_distribution(num_images)
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


def create_compression_file(images, sample_path):
    # sample_path = recreate_folder(sample_path)
    if check_range_sigmoid(images):
        # images = images.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        images = (images * 255).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        np.savez(sample_path, images)
        return sample_path
    else:
        raise ValueError('images must be in range [0, 1]')


def make_jpg_images(tensor, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # If the folder exists, get the highest image index
    existing_images = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    if existing_images:
        last_image_index = max([int(f.split('_')[-1].split('.')[0]) for f in existing_images])
    else:
        last_image_index = -1

    # Convert the tensor elements to PIL Images and save them
    for i in range(len(tensor)):
        image_array = np.uint8(tensor[i].to("cpu").numpy() * 255)  # Rescale pixel values to [0, 255]
        image = Image.fromarray(np.transpose(image_array, (1, 2, 0)))  # Convert to PIL Image
        # Save the image to the output folder with a unique filename
        image_path = os.path.join(output_folder, f'image_{last_image_index + i + 1}.jpg')
        image.save(image_path)

    return output_folder


def zip_images(directory, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, directory))


def check_range_sigmoid(tensor):
    return (tensor >= 0).all() and (tensor <= 1).all()