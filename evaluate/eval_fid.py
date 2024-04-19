import torch

from evaluate.eval_ws import compute_F_AD
from cleanfid import fid
from utils import *
from metrics.wasserstein import *


def compute_WG(generated_images, real_images):
    return fid.compute_fid(generated_images, real_images)


def compute_LP(encoded_samples, prior_samples):
    LP = compute_true_Wasserstein(X=encoded_samples[:10000], Y=prior_samples[:10000])
    return LP


def compute_RL(decoded_images, real_images):
    RL = torch.nn.functional.binary_cross_entropy(decoded_images, real_images)
    return RL


# def compute_F_AD_images(model,
#                         prior_distribution,
#                         tensor_real_images,
#                         tensor_labels,
#                         num_classes,
#                         stat_dir,
#                         device):
#     each_class_images = dict()
#     for cls_id in range(num_classes):
#         if cls_id in each_class_images:
#             each_class_images[cls_id].append(tensor_real_images[tensor_labels == cls_id])
#         else:
#             each_class_images[cls_id] = list()
#             each_class_images[cls_id].append(tensor_real_images[tensor_labels == cls_id])
#
#     list_distance = list()
#     for cls_id, list_real_images in each_class_images.items():
#         real_images = torch.cat(list_real_images, dim=0).cpu()
#         num_images = len(real_images)
#         generated_images = generate_image(model=model,
#                                           prior_distribution=prior_distribution,
#                                           num_images=num_images,
#                                           device=device)
#         npz_cls_images = create_compression_file(real_images, f"{stat_dir}/real_images.npz")
#         npz_gen_images = create_compression_file(generated_images, f"{stat_dir}/generated_images.npz")
#         IS, FID, sFID, precision, recall = fid_evaluator_function(npz_gen_images, npz_cls_images)
#         list_distance.append(FID)
#     return compute_fairness(list_distance), compute_averaging_distance(list_distance)

def compute_F_AD_images(args, gen_path, list_gen_paths):
    list_distance = list()
    for gen_path in list_gen_paths:
        dist = fid.compute_fid(gen_path, gen_path)
        list_distance.append(dist)
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

        list_images_paths = list()

        for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
            list_real_images.append(x_test)
            list_labels.append(y_test)

            decoded_images, encoded_images = model(x_test.to(device))

            list_encoded_images.append(encoded_images.detach())
            list_decoded_images.append(decoded_images.detach())

            for cls_id in range(args.num_classes):
                class_jpg = make_jpg_images(tensor=x_test[y_test == cls_id],
                                            output_folder=f"{args.images_path}/class_{cls_id}")
                list_images_paths.append(class_jpg)

        tensor_real_images = torch.cat(list_real_images, dim=0).cpu()
        tensor_labels = torch.cat(list_labels, dim=0).cpu()
        tensor_encoded_images = torch.cat(list_encoded_images, dim=0).cpu()
        tensor_decoded_images = torch.cat(list_decoded_images, dim=0).cpu()

        num_images = tensor_real_images.shape[0]
        tensor_generated_images = generate_image(model=model,
                                                 prior_distribution=prior_distribution,
                                                 num_images=num_images,
                                                 device=device).cpu()

        generated_images_path = make_jpg_images(tensor=tensor_generated_images,
                                                output_folder=f"{args.gen_dir}/generated_images")
        real_images_path = make_jpg_images(tensor=tensor_real_images, output_folder=f"{args.images_path}/ground_truth")
        list_images_paths.append(generated_images_path)
        list_images_paths.append(real_images_path)

        for image_path in list_images_paths:
            zip_images(images=image_path, zip_filename=image_path)

        device = 'cpu'
        # Compute RL
        RL = compute_RL(tensor_decoded_images, tensor_real_images)
        print(f"RL: {RL}")
        # Compute WG
        WG = compute_WG(generated_images_path, real_images_path)
        print(f"WG: {WG}")
        # Compute LP
        prior_samples = prior_distribution(num_images)
        print(tensor_encoded_images.shape, prior_samples.shape)
        LP = compute_LP(tensor_encoded_images, prior_samples)
        print(f"LP: {LP}")
        # Compute F and AD in latent space
        F, AD = compute_F_AD(list_features=tensor_encoded_images,
                             list_labels=tensor_labels,
                             prior_distribution=prior_distribution,
                             num_classes=args.num_classes,
                             device=device)
        print(f"F: {F}, AD: {AD}")
        # Compute F and AD in image space
        # F_images, AD_images = compute_F_AD_images(model=model,
        #                                           prior_distribution=prior_distribution,
        #                                           tensor_real_images=tensor_real_images,
        #                                           tensor_labels=tensor_labels,
        #                                           num_classes=args.num_classes,
        #                                           stat_dir=image_path,
        #                                           device=device)
        assert len(list_images_paths[:-2]) == args.num_classes
        F_images, AD_images = compute_F_AD_images(args, generated_images_path, list_images_paths[:-2])
        print(f"FI: {F_images}, ADI: {AD_images}")
        RL = convert_to_cpu_number(RL)
        LP = convert_to_cpu_number(LP)
        WG = convert_to_cpu_number(WG)
        F = convert_to_cpu_number(F)
        AD = convert_to_cpu_number(AD)
        F_images = convert_to_cpu_number(F_images)
        AD_images = convert_to_cpu_number(AD_images)

        return RL, LP, WG, F, AD, F_images, AD_images
