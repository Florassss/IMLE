import os
import sys
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data.util import imresize_np
from data import create_dataset, create_dataloader
from models import create_model
from utils.logger import PrintLogger
import torch
import sys
import os.path
import glob
import pickle
import lmdb
import cv2
import math
import numpy as np
from utils.progress_bar import ProgressBar
from dci_util import *

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

# print to file and std_out simultaneously
sys.stdout = PrintLogger(opt['path']['log'])
print('\n**********' + util.get_timestamp() + '**********')


use_dci = False if 'use_dci' not in opt['train'] else opt['train']['use_dci']
use_lpips = False if 'use_lpips' not in opt['train'] else opt['train']['use_lpips']

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    print('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    for data in test_loader:
        need_HR = False
        not_test = True if 'D1' in data else False

        multiple = 1 if "multiple" not in opt else opt["multiple"]

        for run_index in range(multiple):
            if use_dci and not_test:
                cur_month_code = get_code_for_data_three_lpips(model, data, opt)
            else:
                code_val_0 = torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                         data['LR'].shape[2], data['LR'].shape[3])
                code_val_1 = torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                         data['LR'].shape[2] * 2, data['LR'].shape[3] * 2)
                code_val_2 = torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                         data['LR'].shape[2] * 4, data['LR'].shape[3] * 4)
                cur_month_code = [code_val_0, code_val_1, code_val_2]
            model.feed_data(data, code=cur_month_code)

            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            model.test()  # test
            visuals = model.get_current_visuals(need_HR=need_HR)

            if opt['double']:

                code_val = torch.randn(1, int(opt['network_G']['in_code_nc']),
                                       visuals['SR'].shape[1], visuals['SR'].shape[2])

                cur_data = {'LR': visuals['SR'].reshape((1,) + visuals['SR'].shape)}

                model.feed_data(cur_data, code=code_val, need_HR=need_HR)
                model.test()  # test
                visuals = model.get_current_visuals(need_HR=need_HR)

            elif opt['triple']:
                code_val = torch.randn(1, int(opt['network_G']['in_code_nc']),
                                      visuals['SR'].shape[1], visuals['SR'].shape[2])

                cur_data = {'LR': visuals['SR'].reshape((1,) + visuals['SR'].shape)}

                model.feed_data(cur_data, code=code_val, need_HR=need_HR)
                model.test()  # test
                visuals = model.get_current_visuals(need_HR=need_HR)

                code_val = torch.randn(1, int(opt['network_G']['in_code_nc']),
                                      visuals['SR'].shape[1], visuals['SR'].shape[2])

                cur_data = {'LR': visuals['SR'].reshape((1,) + visuals['SR'].shape)}

                model.feed_data(cur_data, code=code_val, need_HR=need_HR)
                model.test()  # test
                visuals = model.get_current_visuals(need_HR=need_HR)

            sr_img = util.tensor2img(visuals['SR'])  # uint8

            if opt["down_scale"]:
                sr_img = imresize_np(sr_img, opt["down_scale"], True)

            if need_HR:  # load GT image and calculate psnr
                gt_img = util.tensor2img(visuals['HR'])
                psnr = util.psnr(sr_img, gt_img)
                ssim = util.ssim(sr_img, gt_img, multichannel=True)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                if gt_img.shape[2] == 3:  # RGB image
                    cropped_sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    cropped_gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    psnr_y = util.psnr(cropped_sr_img_y, cropped_gt_img_y)
                    ssim_y = util.ssim(cropped_sr_img_y, cropped_gt_img_y, multichannel=False)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    print('{:20s} - PSNR: {:.4f} dB; SSIM: {:.4f}; PSNR_Y: {:.4f} dB; SSIM_Y: {:.4f}.'\
                        .format(img_name, psnr, ssim, psnr_y, ssim_y))
                else:
                    print('{:20s} - PSNR: {:.4f} dB; SSIM: {:.4f}.'.format(img_name, psnr, ssim))
            else:
                print(img_name)

            suffix = opt['suffix']
            save_img_path = os.path.join(dataset_dir, img_name + '.png')
            print("saving: %s" % save_img_path)
            util.save_img(sr_img, save_img_path)

    if need_HR:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.4f} dB; SSIM: {:.4f}\n'\
                .format(test_set_name, ave_psnr, ave_ssim))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.4f} dB; SSIM_Y: {:.4f}\n'\
                .format(ave_psnr_y, ave_ssim_y))
