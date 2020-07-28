import imageio
# import cv2
import numpy as np
import glob
import os

# source_dir = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/best_sample/bl_1/*"
# imgs = np.array([imageio.imread(v) for v in sorted(glob.glob(source_dir))])
# for v in sorted(glob.glob(source_dir)):
#     print(v)
# imageio.mimwrite('/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/best_sample/bl_1.mp4', imgs, fps=2)

def merge_videos(p_0, p_1, p_2, save_p, vid_p):
    result = []
    low = 60
    wid = 20
    # cur = np.ones((512 + low, 512 * 3 + wid * 2, 3)) * 255
    list_1 = sorted(glob.glob(p_1))
    list_2 = sorted(glob.glob(p_2))
    for i, v in enumerate(sorted(glob.glob(p_0))):
        cur = np.ones((512 + low, 512 * 3 + wid * 2, 3)) * 255
        cur[0:512, 0:512, :] = imageio.imread(v)
        cur[0:512, (512 + wid):(1024 + wid), :] = imageio.imread(list_1[i])
        cur[0:512, (1024 + wid * 2):, :] = imageio.imread(list_2[i])
        img_name = v.split('/')[-1]
        imageio.imwrite(os.path.join(save_p, img_name), cur)
        result.append(cur)
    print(len(result))
    # imageio.mimwrite(vid_p, np.array(result), fps=2)

def make_video(source_dir, save_p):
    imgs = np.array([imageio.imread(v) for v in sorted(glob.glob(source_dir))])
    imageio.mimwrite(save_p, imgs, fps=2)


# path_1 = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/best_sample/bl_1/*"
# path_2 = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/best_sample/bl_2/*"
# path_3 = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/best_sample/srim/*"
# save_p = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/best_sample/merged/"
video_p = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/variation/var.mp4"
# merge_videos(path_1, path_2, path_3, save_p, video_p)

src_p = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/variation/v/*"
make_video(src_p, video_p)

# path_1 = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/variation/bl_1/*"
# path_2 = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/variation/bl_2/*"
# path_3 = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/variation/srim/*"
# save_p = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/variation/merged/"
# video_p = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/variation/variation.mp4"
# merge_videos(path_1, path_2, path_3, save_p, video_p)

# src_p = "/Users/niopeng/Documents/Research/Berkeley/NeurIPS2020/variation/vid/*"
# make_video(src_p, video_p)


