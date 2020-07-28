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
from dci_util import *
from torch.autograd import Variable

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

device0 = torch.device("cuda:1")
device1 = torch.device("cuda:0")
# Create model
model = create_model(opt)
model1 = create_model(opt1)


first_results = []
use_lpips = False if 'use_lpips' not in opt['train'] else opt['train']['use_lpips']
keep_last = True if 'keep_last' not in opt else opt['keep_last']
multiple = 1 if "multiple" not in opt else opt["multiple"]

for phase, dataset_opt in sorted(opt1['datasets'].items()):
    test_set1 = create_dataset(dataset_opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    print('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    for data in test_loader:
        need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

        for run_index in range(multiple):
            code_val_0 = Variable(torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                  data['LR'].shape[2], data['LR'].shape[3], device=device0), requires_grad=True)
            code_val_1 = Variable(torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                  data['LR'].shape[2] * 2, data['LR'].shape[3] * 2, device=device1), requires_grad=True)
            code_val_2 = Variable(torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                  data['LR'].shape[2] * 4, data['LR'].shape[3] * 4, device=device1), requires_grad=True)
            code_val_3 = Variable(torch.randn(data['LR'].shape[0], int(opt['network_G']['in_code_nc']), 256, 256, device=device1),
                                  requires_grad=True)
            optimizer = torch.optim.Adam([code_val_0, code_val_1, code_val_2], lr=1e-3, betas=(0.9, 0.999))
            optimizer1 = torch.optim.Adam([code_val_3], lr=1e-3, betas=(0.9, 0.999))

            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            target = test_set1[0]['HR']

            for i in range(5000):
                cur_out = model.netG(data['LR'], code_val_0, code_val_1, code_val_2)[-1]
                copied_cur_out = Variable(cur_out.detach().to(device1), requires_grad=True)
                output = model1.netG(copied_cur_out, code_val_3, None, None)[-1]
                dist = model1.loss_fn.forward(output, target, normalize=True)
                optimizer.zero_grad()
                optimizer1.zero_grad()
                dist.backward()
                cur_out.backward(copied_cur_out.grad)
                optimizer1.step()
                optimizer.step()
                if i % 10 == 0:
                    print('iter %d, dist %.3g' % (i, dist.view(-1).data.cpu().numpy()[0]))
                if i % 100 == 0:
                    save_img_path = os.path.join(dataset_dir, img_name + '_%d.png' % i)
                    sr_img = util.tensor2img(output.detach()[0].float().cpu())  # uint8
                    print("saving: %s" % save_img_path)
                    util.save_img(sr_img, save_img_path)
