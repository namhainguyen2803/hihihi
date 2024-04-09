import torch

from evaluate.eval_ws import compute_F_AD
from fid.fid_evaluator import fid_evaluator_function
from utils import *


def compute_WG(generated_images, real_images):
    IS, FID, sFID, precision, recall = fid_evaluator_function(generated_images, real_images)
    return FID


def compute_LP(encoded_samples, prior_samples):
    LP = compute_true_Wasserstein(X=encoded_samples, Y=prior_samples)
    return LP


def compute_RL(decoded_images, real_images):
    RL = torch.nn.functional.binary_cross_entropy(decoded_images, real_images)
    return RL


def compute_F_AD_images(model,
                        prior_distribution,
                        tensor_real_images,
                        tensor_labels,
                        num_classes,
                        device):
    each_class_images = dict()
    for cls_id in range(num_classes):
        if cls_id in each_class_images:
            each_class_images[cls_id].append(tensor_real_images[tensor_labels == cls_id])
        else:
            each_class_images[cls_id] = list()
            each_class_images[cls_id].append(tensor_real_images[tensor_labels == cls_id])

    list_distance = list()
    for cls_id, list_real_images in each_class_images.items():
        real_images = torch.cat(list_real_images, dim=0).cpu()
        num_images = len(real_images)
        generated_images = generate_image(model=model,
                                          prior_distribution=prior_distribution,
                                          num_images=num_images,
                                          device=device)
        npz_cls_images = create_compression_file(real_images, f"statistic/real_images.npz")
        npz_gen_images = create_compression_file(generated_images, f"statistic/generated_images.npz")
        IS, FID, sFID, precision, recall = fid_evaluator_function(npz_gen_images, npz_cls_images)
        list_distance.append(FID)
    return compute_fairness(list_distance), compute_averaging_distance(list_distance)


def ultimate_evaluate_fid(args,
                          model,
                          test_loader,
                          prior_distribution,
                          device):
    with torch.no_grad():
        model.eval()

        list_real_images = list()
        list_labels = list()
        list_encoded_images = list()
        list_decoded_images = list()

        for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
            list_real_images.append(x_test)
            list_labels.append(y_test)

            decoded_images, encoded_images = model(x_test.to(device))

            list_encoded_images.append(encoded_images.detach())
            list_decoded_images.append(decoded_images.detach())

        tensor_real_images = torch.cat(list_real_images, dim=0).cpu()
        tensor_labels = torch.cat(list_labels, dim=0).cpu()
        tensor_encoded_images = torch.cat(list_encoded_images, dim=0).cpu()
        tensor_decoded_images = torch.cat(list_decoded_images, dim=0).cpu()

        num_images = tensor_real_images.shape[0]
        tensor_generated_images = generate_image(model=model,
                                                 prior_distribution=prior_distribution,
                                                 num_images=num_images,
                                                 device=device).cpu()
        print(tensor_generated_images.shape, tensor_generated_images.shape)
        npz_real_images = create_compression_file(tensor_real_images, "statistic/real_images.npz")
        npz_generated_images = create_compression_file(tensor_generated_images, "statistic/generated_images.npz")

        device = 'cpu'

        # Compute RL
        RL = compute_RL(tensor_decoded_images, tensor_real_images)

        # Compute WG
        WG = compute_WG(npz_generated_images, npz_real_images)

        # Compute LP
        prior_samples = prior_distribution(num_images).to(device)
        LP = compute_LP(tensor_encoded_images, prior_samples)

        # Compute F and AD in latent space
        F, AD = compute_F_AD(list_features=tensor_encoded_images,
                             list_labels=tensor_labels,
                             prior_distribution=prior_distribution,
                             num_classes=args.num_classes,
                             device=device)

        # Compute F and AD in image space
        F_images, AD_images = compute_F_AD_images(model=model,
                                                  prior_distribution=prior_distribution,
                                                  tensor_real_images=tensor_real_images,
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
