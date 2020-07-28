import argparse
import utils.util as util
import cv2
import glob


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(path)
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, ::-1]


def get_sim(root_path, ori_path, n_samples):
    img_list = sorted(glob.glob(ori_path))
    total_psnr = 0.
    total_ssim = 0.
    count = 0
    for i, v in enumerate(img_list):
        img_name = v.split("/")[-1].split(".")[0]
        img0_np = load_image(v)
        for j in range(n_samples):
            img1_np = load_image(root_path + img_name + "_" + str(j) + ".png")
            total_psnr += util.psnr(img0_np, img1_np)
            total_ssim += util.ssim(img0_np, img1_np, multichannel=True)
            count += 1

    return total_psnr / count, total_ssim / count


n_samples = 50

parser = argparse.ArgumentParser()
parser.add_argument('-hrim', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-gt', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-bl', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-bll', type=str, required=True, help='Path to options JSON file.')

srim_root_path = parser.parse_args().hrim
srim_ori_path = parser.parse_args().gt + "*"
c_psnr, c_ssim = get_sim(srim_root_path, srim_ori_path, n_samples)
print("HyperRIM PSNR: %f, SSIM: %f" % (c_psnr, c_ssim))

bl1_root_path = parser.parse_args().bl
c_psnr, c_ssim = get_sim(bl1_root_path, srim_ori_path, n_samples)
print("Baseline 1 PSNR: %f, SSIM: %f" % (c_psnr, c_ssim))


bl2_root_path = parser.parse_args().bll
c_psnr, c_ssim = get_sim(bl2_root_path, srim_ori_path, n_samples)
print("Baseline 2 PSNR: %f, SSIM: %f" % (c_psnr, c_ssim))


