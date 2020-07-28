import functools
import torch
import torch.nn as nn
from torch.nn import init

import models.modules.architecture as arch

####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'PEOneIMRRDB_net':
        upsample_kernel_mode = "nearest" if "upsample_kernel_mode" not in opt_net else opt_net["upsample_kernel_mode"]
        use_wn = False if 'use_wn' not in opt_net else opt_net['use_wn']
        L = 10 if 'L' not in opt_net else opt_net['L']
        netG = arch.PEOneIMRRDBNet(in_nc=opt_net['in_nc'], in_code_nc=opt_net['in_code_nc'],
                                   out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                   gc=opt_net['gc'], norm_type=opt_net['norm_type'], act_type='leakyrelu',
                                   mode=opt_net['mode'], upsample_kernel_mode=upsample_kernel_mode,
                                   use_sigmoid=opt_net["use_sigmoid"], last_act=opt_net['last_act'],
                                   use_wn=use_wn, L=L)

    elif which_model == 'OneIMRRDB_net':
        upsample_kernel_mode = "nearest" if "upsample_kernel_mode" not in opt_net else opt_net["upsample_kernel_mode"]
        use_wn = False if 'use_wn' not in opt_net else opt_net['use_wn']
        netG = arch.OneIMRRDBNet(in_nc=opt_net['in_nc'], in_code_nc=opt_net['in_code_nc'], out_nc=opt_net['out_nc'],
                                 nf=opt_net['nf'], nb=opt_net['nb'], gc=opt_net['gc'], norm_type=opt_net['norm_type'],
                                 act_type='leakyrelu', mode=opt_net['mode'], upsample_kernel_mode=upsample_kernel_mode,
                                 use_sigmoid=opt_net["use_sigmoid"], last_act=opt_net['last_act'], use_wn=use_wn)

    elif which_model == 'PESkipTwoIMRRDB_net':
        upsample_kernel_mode = "nearest" if "upsample_kernel_mode" not in opt_net else opt_net["upsample_kernel_mode"]
        use_wn = False if 'use_wn' not in opt_net else opt_net['use_wn']
        gcs = [] if "gcs" not in opt_net else opt_net['gcs']
        nfs = [] if "nfs" not in opt_net else opt_net['nfs']
        L = 10 if 'L' not in opt_net else opt_net['L']
        netG = arch.PESkipTwoIMRRDBNet(in_nc=opt_net['in_nc'], in_code_nc=opt_net['in_code_nc'],
                                       out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                       gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
                                       act_type='leakyrelu', mode=opt_net['mode'],
                                       upsample_kernel_mode=upsample_kernel_mode,
                                       use_sigmoid=opt_net["use_sigmoid"], last_act=opt_net['last_act'],
                                       use_wn=use_wn, gcs=gcs, nfs=nfs, L=L)

    elif which_model == 'PESkipThreeIMRRDB_net':  # RRDB
        upsample_kernel_mode = "nearest" if "upsample_kernel_mode" not in opt_net else opt_net["upsample_kernel_mode"]
        use_wn = False if 'use_wn' not in opt_net else opt_net['use_wn']
        gcs = [] if "gcs" not in opt_net else opt_net['gcs']
        nfs = [] if "nfs" not in opt_net else opt_net['nfs']
        L = 10 if 'L' not in opt_net else opt_net['L']
        device = torch.device('cuda') if 'device' not in opt_net else torch.device(opt_net['device'])
        netG = arch.PESkipThreeIMRRDBNet(in_nc=opt_net['in_nc'], in_code_nc=opt_net['in_code_nc'],
                                         out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], gc=opt_net['gc'],
                                         upscale=opt_net['scale'], norm_type=opt_net['norm_type'], act_type='leakyrelu',
                                         mode=opt_net['mode'], upsample_kernel_mode=upsample_kernel_mode,
                                         use_sigmoid=opt_net["use_sigmoid"], last_act=opt_net['last_act'],
                                         use_wn=use_wn, gcs=gcs, nfs=nfs, L=L, device=device)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train']:
        scale = 0.1 if 'init_scale' not in opt_net else opt_net['init_scale']
        init_type = 'kaiming' if 'init_type' not in opt_net else opt_net['init_type']
        init_weights(netG, init_type=init_type, scale=scale)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG, device_ids=gpu_ids) if len(gpu_ids) > 1 else nn.DataParallel(netG)
    return netG
