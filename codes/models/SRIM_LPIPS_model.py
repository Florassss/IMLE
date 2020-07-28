import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
import models.LPIPSmodels as Lmodels


class SRIMLPIPSModel(BaseModel):
    def __init__(self, opt):
        super(SRIMLPIPSModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        self.loss_fn = Lmodels.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
        if self.is_train:
            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            fix_stack = train_opt['fix_stack']
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    if fix_stack:
                        if "stack_%d"%fix_stack not in k:
                            optim_params.append(v)
                    else:
                        optim_params.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_HR=True, code=[]):
        # LR
        self.var_L = data['LR'].to(self.device)
        self.code = code[0].to(self.device) if len(code) > 0 else None
        self.code_1 = code[1].to(self.device) if len(code) > 1 else None
        self.code_2 = code[2].to(self.device) if len(code) > 2 else None
        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)
            self.var_D1 = data['D1'].to(self.device) if 'D1' in data else None
            self.var_D2 = data['D2'].to(self.device) if 'D2' in data else None

    def get_loss(self, level=-1):
        self.netG.eval()
        with torch.no_grad():
            gen_imgs = self.netG(self.var_L, self.code, self.code_1, self.code_2)
            if len(gen_imgs) > 2:
                gts = [self.var_D1, self.var_D2, self.var_H]
            elif len(gen_imgs) > 1:
                gts = [self.var_D1, self.var_H]
            else:
                gts = [self.var_H]
            result = self.loss_fn.forward(gen_imgs[level], gts[level], normalize=True).float().cpu().detach()
        self.netG.train()
        return result

    def optimize_parameters(self, step):
        torch.autograd.set_detect_anomaly(True)
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L, self.code, self.code_1, self.code_2)[-1]
        l_g_total = self.loss_fn.forward(self.fake_H, self.var_H, normalize=True)
        l_g_total.backward()
        self.optimizer_G.step()
        self.log_dict['l_g_lpips'] = l_g_total.item()

    def test(self, keep_last=True):
        self.netG.eval()
        with torch.no_grad():
            output = self.netG(self.var_L, self.code, self.code_1, self.code_2)
            if isinstance(output, list) and keep_last:
                self.fake_H = output[-1]
            else:
                self.fake_H = output
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        if isinstance(self.fake_H, list):
            for i in range(len(self.fake_H) - 1):
                out_dict['SR_%d' % i] = self.fake_H[i].detach()[0].float().cpu()
            out_dict['SR'] = self.fake_H[-1].detach()[0].float().cpu()
        else:
            out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            if 'load_partial' in self.opt and self.opt['load_partial']:
                self.load_partial_network(load_path_G, self.netG)
            else:
                self.load_network(load_path_G, self.netG)

    def load_partial_network(self, load_path, network, strict=False):
        if isinstance(network, nn.DataParallel):
            network = network.module
        pretrained_dict = torch.load(load_path)
        model_dict = network.state_dict()
        for k, v in model_dict.items():
            if k in pretrained_dict:
                model_dict.update({k: pretrained_dict[k]})
        network.load_state_dict(pretrained_dict, strict=strict)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
