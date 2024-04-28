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


def compute_fairness_and_averaging_distance_in_images_space(model,
                                                            tensor_flatten_real_images,
                                                            prior_distribution,
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
        flatten_real_images = torch.cat(list_flatten_real_images, dim=0)
        num_images_class = flatten_real_images.shape[0]
        tensor_generated_images = generate_image(model=model,
                                                 prior_distribution=prior_distribution,
                                                 num_images=num_images_class,
                                                 device=device)
        tensor_flatten_generated_images = tensor_generated_images.reshape(num_images_class, -1)
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
        model.to("cpu")
        
        list_labels = list()
        list_encoded_images = list()
        list_real_images = list()
        RL = 0
        total_images = 0

        for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
            list_real_images.append(x_test)
            list_labels.append(y_test)
            num_images = x_test.shape[0]
            total_images += num_images
            decoded_images, encoded_images = model(x_test)
            list_encoded_images.append(encoded_images.detach())
            RL += torch.nn.functional.binary_cross_entropy(x_test, decoded_images) * num_images

        tensor_real_images = torch.cat(list_real_images, dim=0)
        tensor_labels = torch.cat(list_labels, dim=0)
        tensor_encoded_images = torch.cat(list_encoded_images, dim=0)

        print(f"Number of images in testing: {total_images}")
        # Compute RL
        RL = RL / total_images
        print(RL)
        
        # Compute WG
        WG = 0
        WG = compute_true_Wasserstein(X=tensor_flatten_generated_images, Y=tensor_flatten_real_images)
        print(WG)
        
        # Compute LP
        LP = 0
        prior_samples = prior_distribution(num_images)
        LP = compute_true_Wasserstein(X=tensor_encoded_images, Y=prior_samples)
        print(LP)
        
        # Compute F and AD in latent space
        F, AD = compute_F_AD(list_features=tensor_encoded_images,
                             list_labels=tensor_labels,
                             prior_distribution=prior_distribution,
                             num_classes=args.num_classes,
                             device="cpu")
        print(F, AD)
        
        # Compute F and AD in image space
        F_images, AD_images = compute_fairness_and_averaging_distance_in_images_space(model=model,
                                                                                      tensor_flatten_real_images=tensor_real_images.reshape(total_images, -1),
                                                                                      prior_distribution=prior_distribution,
                                                                                      tensor_labels=tensor_labels,
                                                                                      num_classes=args.num_classes,
                                                                                      device="cpu")
        print(F_images, AD_images)
        RL = convert_to_cpu_number(RL)
        LP = convert_to_cpu_number(LP)
        WG = convert_to_cpu_number(WG)
        F = convert_to_cpu_number(F)
        AD = convert_to_cpu_number(AD)
        F_images = convert_to_cpu_number(F_images)
        AD_images = convert_to_cpu_number(AD_images)
        
        model.to(device)
        return RL, LP, WG, F, AD, F_images, AD_images
