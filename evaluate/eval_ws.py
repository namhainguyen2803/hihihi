from metrics.wasserstein import *
import matplotlib.pyplot as plt
from utils import generate_image, compute_fairness, compute_averaging_distance, convert_to_cpu_number


def compute_F_AD(list_features,
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
            dist_swd.append(wd)
    return compute_fairness(dist_swd), compute_averaging_distance(dist_swd)


def compute_fairness_and_averaging_distance_in_images_space(tensor_flatten_generated_images,
                                                            tensor_flatten_real_images,
                                                            tensor_labels,
                                                            num_classes):
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
        ws = compute_true_Wasserstein(X=tensor_flatten_generated_images, Y=flatten_real_images)
        list_ws_distance.append(ws)

    return compute_fairness(list_ws_distance), compute_averaging_distance(list_ws_distance)


def ultimate_evaluation(args,
                        model,
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

            decoded_images, encoded_images = model(x_test.to(device))

            list_encoded_images.append(encoded_images.detach())
            list_decoded_images.append(decoded_images.detach())

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
        F, AD = compute_F_AD(list_features=tensor_encoded_images,
                             list_labels=tensor_labels,
                             prior_distribution=prior_distribution,
                             num_classes=args.num_classes,
                             device=device)

        # Compute F and AD in image space
        F_images, AD_images = compute_fairness_and_averaging_distance_in_images_space(tensor_flatten_generated_images=tensor_flatten_generated_images,
                                                                                      tensor_flatten_real_images=tensor_flatten_real_images,
                                                                                      tensor_labels=tensor_labels,
                                                                                      num_classes=args.num_classes)

        RL = convert_to_cpu_number(RL)
        LP = convert_to_cpu_number(LP)
        WG = convert_to_cpu_number(WG)
        F = convert_to_cpu_number(F)
        AD = convert_to_cpu_number(AD)
        F_images = convert_to_cpu_number(F_images)
        AD_images = convert_to_cpu_number(AD_images)

        return RL, LP, WG, F, AD, F_images, AD_images
