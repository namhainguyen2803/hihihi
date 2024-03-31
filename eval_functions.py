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


def compute_fairness_and_averaging_distance_in_images_space(model,
                                                            prior_distribution,
                                                            tensor_flatten_real_images,
                                                            tensor_labels,
                                                            num_projections,
                                                            num_classes,
                                                            device,
                                                            theta):
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

        ws = sliced_wasserstein_distance(encoded_samples=flatten_real_images,
                                         distribution_samples=flatten_generated_images,
                                         num_projections=num_projections,
                                         p=2,
                                         device='cpu',
                                         theta=theta)

        list_ws_distance.append(ws)

    return compute_fairness(list_ws_distance), compute_averaging_distance(list_ws_distance)


def ultimate_evaluation(args, model, evaluator, test_loader, prior_distribution, theta=None, theta_latent=None,
                        device='cpu'):
    with torch.no_grad():
        model.eval()

        list_real_images = list()
        list_labels = list()
        list_encoded_images = list()
        list_decoded_images = list()

        each_class_images = dict()

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
        print(f"Reconstruction loss: {RL}")

        # Compute WG
        if theta is None:
            theta = rand_projections(dim=tensor_flatten_real_images.shape[-1],
                                     num_projections=args.num_projections,
                                     device=device)

        WG = sliced_wasserstein_distance(encoded_samples=tensor_flatten_generated_images,
                                         distribution_samples=tensor_flatten_real_images,
                                         num_projections=args.num_projections,
                                         p=2,
                                         device=device,
                                         theta=theta)

        print(f"Wasserstein distance between generated and real images (WG): {WG}")

        # Compute LP
        prior_samples = prior_distribution(num_images).to(device)
        if theta_latent is None:
            theta_latent = rand_projections(dim=tensor_encoded_images.shape[-1],
                                            num_projections=args.num_projections,
                                            device=device)

        LP = sliced_wasserstein_distance(encoded_samples=tensor_encoded_images,
                                         distribution_samples=prior_samples,
                                         num_projections=args.num_projections,
                                         p=2,
                                         device=device,
                                         theta=theta_latent)

        print(f"Wasserstein distance between posterior and prior distribution (LP): {LP}")

        # Compute F and AD in latent space
        if theta_latent is None:
            theta_latent = rand_projections(dim=tensor_encoded_images.shape[-1],
                                            num_projections=args.num_projections,
                                            device=device)

        F, AD = compute_fairness_and_averaging_distance(list_features=tensor_encoded_images,
                                                        list_labels=tensor_labels,
                                                        prior_distribution=prior_distribution,
                                                        num_classes=args.num_classes,
                                                        device=device,
                                                        num_projections=args.num_projections,
                                                        dim=tensor_encoded_images.shape[-1],
                                                        theta=theta_latent)

        print(f"Fairness (F): {F}")
        print(f"Averaging distance (AD): {AD}")

        ### Compute F and AD in image spaces
        if theta is None:
            theta = rand_projections(dim=tensor_flatten_real_images.shape[-1],
                                     num_projections=args.num_projections,
                                     device=device)

        F_images, AD_images = compute_fairness_and_averaging_distance_in_images_space(model=model,
                                                                                      prior_distribution=prior_distribution,
                                                                                      tensor_flatten_real_images=tensor_flatten_real_images,
                                                                                      tensor_labels=tensor_labels,
                                                                                      num_projections=args.num_projections,
                                                                                      num_classes=args.num_classes,
                                                                                      device=device,
                                                                                      theta=theta)

        print(f"Fairness in images space (FI): {F_images}")
        print(f"Averaging distance in images space (ADI): {AD_images}")

        return RL, LP, WG, F, AD, F_images, AD_images
