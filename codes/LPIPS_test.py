import argparse
import os
import torch
# from IPython import embed
# from util import util
import numpy as np
import models.LPIPSmodels as Lmodels
import cv2
import glob


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(path)
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, ::-1]


def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def add_dict(map, key, val):
    if key in map:
        map[key].append(val)
    else:
        map[key] = [val]


def get_weights(root_path, ori_path, n_samples):
    img_list = sorted(glob.glob(ori_path))
    data_length = len(img_list)
    result = np.empty((data_length * n_samples))
    mapping = {}

    for i, v in enumerate(img_list):
        img_name = v.split("/")[-1].split(".")[0]
        img0_np = load_image(v)
        img0 = im2tensor(img0_np)

        for j in range(n_samples):
            img1_np = load_image(root_path + img_name + "_" + str(j) + ".png")
            img1 = im2tensor(img1_np)
            result[i * n_samples + j] = model.forward(img1, img0).view(-1).data.cpu().numpy()[0]
            add_dict(mapping, img_name, img1_np)
    return result, mapping


def compute_weight(diff, sigma):
    return np.exp(-diff / (2 * (sigma ** 2)))


def compute_mean(mapping):
    result = {}
    for key, val in mapping.items():
        all_img = np.stack(val, axis=0)
        result[key] = np.mean(all_img, axis=0)
    return result


def get_weighted_score(root_path, ori_path, n_samples, dists, sigma, mean_mapping):
    img_list = sorted(glob.glob(ori_path))

    total_var_result = 0

    lp_total = 0

    for i, v in enumerate(img_list):
        count = 0

        img_name = v.split("/")[-1].split(".")[0]
        for j in range(n_samples):
            img1_np = cv2.imread(root_path + img_name + "_" + str(j) + ".png")

            cur_weight = compute_weight(dists[i * n_samples + j], sigma)

            count += cur_weight

            lp_total += cur_weight * model.forward(im2tensor(img1_np), im2tensor(mean_mapping[img_name])).view(-1).data.cpu().numpy()[0]

            total_var_result += cur_weight * np.sum(np.linalg.norm(img1_np - mean_mapping[img_name]))

    return total_var_result / (len(img_list) * n_samples), lp_total / (len(img_list) * n_samples)


## Initializing the model
model = Lmodels.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
n_samples = 50

# srim
os.system("nvidia-smi")

parser = argparse.ArgumentParser()
parser.add_argument('-hrim', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-gt', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-bl', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-bll', type=str, required=True, help='Path to options JSON file.')

srim_root_path = parser.parse_args().hrim
srim_ori_path = parser.parse_args().gt + "*"


srim_dists, srim_mapping = get_weights(srim_root_path, srim_ori_path, n_samples)
srim_mean = compute_mean(srim_mapping)
print("SRIM dist max: %f, min: %f, avg: %f" % (srim_dists.max(), srim_dists.min(), np.mean(srim_dists)))

bl1_root_path = parser.parse_args().bl
bl1_ori_path = srim_ori_path

bl1_dists, bl1_mapping = get_weights(bl1_root_path, bl1_ori_path, n_samples)
bl1_mean = compute_mean(bl1_mapping)
print("Baseline 1 dist max: %f, min: %f, mean: %f" % (bl1_dists.max(), bl1_dists.min(), np.mean(bl1_dists)))

bl2_root_path = parser.parse_args().bll
bl2_ori_path = srim_ori_path

bl2_dists, bl2_mapping = get_weights(bl2_root_path, bl2_ori_path, n_samples)
bl2_mean = compute_mean(bl2_mapping)
print("Baseline 2 dist max: %f, min: %f, mean: %f" % (bl2_dists.max(), bl2_dists.min(), np.mean(bl2_dists)))
print("\n")

# compute sigma
all_dists = np.concatenate((srim_dists, bl1_dists, bl2_dists))
sigmas = [0.3, 0.2, 0.15]


for sigma in sigmas:
    # compute srim score
    srim_score, srim_lp_score = get_weighted_score(srim_root_path, srim_ori_path, n_samples, srim_dists, sigma, srim_mean)
    print("Sigma: %f, SRIM score: %f, LPIPS score: %f"%(sigma, srim_score, srim_lp_score))
    #
    bl1_score, bl1_lp_score = get_weighted_score(bl1_root_path, bl1_ori_path, n_samples, bl1_dists, sigma, bl1_mean)
    print("Sigma: %f, Baseline 1 score: %f, LPIPS score: %f" % (sigma, bl1_score, bl1_lp_score))

    bl2_score, bl2_lp_score = get_weighted_score(bl2_root_path, bl2_ori_path, n_samples, bl2_dists, sigma, bl2_mean)
    print("Sigma: %f, Baseline 2 score: %f, LPIPS score: %f"%(sigma, bl2_score, bl2_lp_score))
    print("\n")


