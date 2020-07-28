import os
import sys
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from utils.logger import PrintLogger
import torch
import sys
import os.path

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-opts', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt1 = option.parse(parser.parse_args().opts, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)
opt1 = option.dict_to_nonedict(opt1)

# print to file and std_out simultaneously
sys.stdout = PrintLogger(opt['path']['log'])
print('\n**********' + util.get_timestamp() + '**********')

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

first_results = []
use_lpips = False if 'use_lpips' not in opt['train'] else opt['train']['use_lpips']
keep_last = True if 'keep_last' not in opt else opt['keep_last']
multiple = 1 if "multiple" not in opt else opt["multiple"]

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
        need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

        for run_index in range(multiple):
            code_val_0 = torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                     data['LR'].shape[2], data['LR'].shape[3])
            code_val_1 = torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                     data['LR'].shape[2] * 2, data['LR'].shape[3] * 2)
            code_val_2 = torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                     data['LR'].shape[2] * 4, data['LR'].shape[3] * 4)
            model.feed_data(data, code=[code_val_0, code_val_1, code_val_2])

            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            model.test(keep_last=keep_last)  # test
            visuals = model.get_current_visuals(need_HR=need_HR)

            if 'SR_1' in visuals:
                save_img_path_0 = os.path.join(dataset_dir, img_name + '_%d_s_2.png' % run_index)
                save_img_path_1 = os.path.join(dataset_dir, img_name + '_%d_s_3.png' % run_index)
                stack_2 = util.tensor2img(visuals['SR_1'])
                stack_3 = util.tensor2img(visuals['SR'])
                util.save_img(stack_2, save_img_path_0)
                util.save_img(stack_3, save_img_path_1)

            gen_data = {
                'LR': torch.unsqueeze(visuals['SR'], 0),
                'LR_path': data['LR_path'],
                'HR': data['HR']
            }
            first_results.append(gen_data)
            sr_img = util.tensor2img(visuals['SR'])  # uint8

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

model = create_model(opt1)

for phase, dataset_opt in sorted(opt1['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))

test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []
test_results['psnr_y'] = []
test_results['ssim_y'] = []
total_loss = 0
avg_lips = 0.0
cur_lpips = 0.0

for i, lr in enumerate(first_results):
    need_HR = True

    code_val_0 = torch.randn(lr['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                             lr['LR'].shape[2], lr['LR'].shape[3])

    lr['HR'] = test_set[int(i // multiple)]['HR']

    model.feed_data(lr, code=[code_val_0])

    img_path = lr['LR_path'][0]
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    model.test()  # test
    visuals = model.get_current_visuals(need_HR=need_HR)

    sr_img = util.tensor2img(visuals['SR'])  # uint8

    if use_lpips:
        cur_lpips = torch.sum(model.get_loss(level=-1))
        avg_lips += cur_lpips

    if need_HR:
        gt_img = util.tensor2img(lr['HR'])

        cropped_sr_img = sr_img
        cropped_gt_img = gt_img
        psnr = util.psnr(cropped_sr_img, cropped_gt_img)
        ssim = util.ssim(cropped_sr_img, cropped_gt_img, multichannel=True)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        print('{:20s} - PSNR: {:.4f} dB; SSIM: {:.4f}; LIPS: {:.4f}.'.format(img_name, psnr, ssim, cur_lpips))
    else:
        print(img_name)
    suffix = opt['suffix']
    save_img_path = os.path.join(dataset_dir, img_name + '_%d.png' % (int(i % multiple)))
    print("saving: %s" % save_img_path)
    util.save_img(sr_img, save_img_path)

if need_HR:  # metrics
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    avg_lips = avg_lips / len(first_results)
    print('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.4f} dB; SSIM: {:.4f}; LIPS: {:.4f}\n'\
            .format(test_set_name, ave_psnr, ave_ssim, avg_lips))
