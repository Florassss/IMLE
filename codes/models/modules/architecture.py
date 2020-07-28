import math
import torch
import torch.nn as nn
from . import block as B


####################
# Generator
####################
class PEOneIMRRDBNet(nn.Module):
    def __init__(self, in_nc, in_code_nc, out_nc, nf, nb, gc=32, norm_type=None, act_type='leakyrelu', mode='CNA',
                 upsample_kernel_mode="nearest", use_sigmoid=False, last_act=None, use_wn=False, L=10):
        super(PEOneIMRRDBNet, self).__init__()

        self.num_stacks = 1

        self.fea_conv = B.conv_block(in_nc + in_code_nc, nf, kernel_size=3, norm_type=None, act_type=None,
                                     use_wn=use_wn)
        rb_blocks = [B.RRDB(nf + 4 * L, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero',
                            norm_type=norm_type, act_type=act_type, mode='CNA', use_wn=use_wn) for _ in range(nb)]
        LR_conv = B.conv_block(nf + 4 * L, nf + 4 * L, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode,
                               use_wn=use_wn)

        HR_conv0 = B.conv_block(nf + 4 * L, nf, kernel_size=3, norm_type=None, act_type=act_type, use_wn=use_wn)
        if use_sigmoid:
            HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type="sigmoid", use_wn=use_wn)
        else:
            HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=last_act, use_wn=use_wn)

        self.stack_1 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                    B.upconv_block(nf + 4 * L, nf + 4 * L, act_type=act_type,
                                                   mode=upsample_kernel_mode),
                                    HR_conv0, HR_conv1)

        self.last_act = last_act

        self.L = L

    def gen_pe(self, inp):
        if not hasattr(self, 'w_pos'):
            bs, w, h = inp.size(0), inp.size(2), inp.size(3)
            w_pos = torch.linspace(-1, 1, w).reshape(1, w).expand(w, -1)
            h_pos = torch.linspace(-1, 1, h).reshape(h, 1).expand(-1, h)
            periodic_fns = [torch.sin, torch.cos]
            freq_bands = 2. ** torch.arange(self.L) * math.pi
            embed_fns = []
            for freq in freq_bands:
                for p_fn in periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            self.w_pe = torch.stack([embed_fn(w_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(torch.device('cuda'))
            self.h_pe = torch.stack([embed_fn(h_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(torch.device('cuda'))
        return torch.cat((inp, self.w_pe, self.h_pe), dim = 1)

    def forward(self, img, code, code_1=None, code_2=None):

        x_1 = torch.cat((img, code), dim=1)
        x_1 = self.fea_conv(x_1)
        x_1 = self.stack_1(self.gen_pe(x_1))
        if self.last_act == "tanh":
            new_result = x_1 + 1
            new_result /= 2.
            x_1 = new_result
        return [x_1]


class OneIMRRDBNet(nn.Module):
    def __init__(self, in_nc, in_code_nc, out_nc, nf, nb, gc=32, norm_type=None, act_type='leakyrelu', mode='CNA',
                 upsample_kernel_mode="nearest", use_sigmoid=False, last_act=None, use_wn=False):
        super(OneIMRRDBNet, self).__init__()

        self.num_stacks = 1

        fea_conv = B.conv_block(in_nc + in_code_nc, nf, kernel_size=3, norm_type=None, act_type=None, use_wn=use_wn)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero',
                            norm_type=norm_type, act_type=act_type, mode='CNA', use_wn=use_wn) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode, use_wn=use_wn)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type, use_wn=use_wn)
        if use_sigmoid:
            HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type="sigmoid", use_wn=use_wn)
        else:
            HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=last_act, use_wn=use_wn)

        self.stack_1 = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                   B.upconv_block(nf, nf, act_type=act_type, mode=upsample_kernel_mode), HR_conv0,
                                   HR_conv1)

        self.last_act = last_act

    def forward(self, img, code, code_1=None, code_2=None):
        x_1 = torch.cat((img, code), dim=1)
        x_1 = self.stack_1(x_1)
        if self.last_act == "tanh":
            new_result = x_1 + 1
            new_result /= 2.
            x_1 = new_result
        return [x_1]


class PESkipTwoIMRRDBNet(nn.Module):
    def __init__(self, in_nc, in_code_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu',
                 mode='CNA', upsample_kernel_mode="nearest", use_sigmoid=False, last_act=None, use_wn=False,
                 gcs=[], nfs=[], L=[]):
        super(PESkipTwoIMRRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        self.num_stacks = n_upscale
        cur_gc = gc if len(gcs) < 1 else gcs[0]
        cur_nf = nf if len(nfs) < 1 else nfs[0]

        self.fea_conv1 = B.conv_block(in_nc + in_code_nc, cur_nf, kernel_size=3, norm_type=None, act_type=None,
                                      use_wn=use_wn)
        rb_blocks = [B.RRDB(cur_nf + 4 * L, kernel_size=3, gc=cur_gc, stride=1, bias=True, pad_type='zero',
                            norm_type=norm_type, act_type=act_type, mode='CNA', use_wn=use_wn) for _ in range(nb)]
        LR_conv = B.conv_block(cur_nf + 4 * L, cur_nf + 4 * L, kernel_size=3, norm_type=norm_type, act_type=None,
                               mode=mode,
                               use_wn=use_wn)

        HR_conv0 = B.conv_block(cur_nf + 4 * L, cur_nf, kernel_size=3, norm_type=None, act_type=act_type, use_wn=use_wn)
        if use_sigmoid:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type="sigmoid", use_wn=use_wn)
        else:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type=last_act, use_wn=use_wn)

        self.stack_1 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                    B.upconv_block(cur_nf + 4 * L, cur_nf + 4 * L, act_type=act_type,
                                                   mode=upsample_kernel_mode), HR_conv0, HR_conv1)

        cur_gc = gc if len(gcs) < 1 else gcs[1]
        cur_nf = nf if len(nfs) < 1 else nfs[1]

        self.fea_conv2 = B.conv_block(in_nc + in_code_nc, cur_nf, kernel_size=3, norm_type=None, act_type=None,
                                      use_wn=use_wn)
        rb_blocks = [B.RRDB(cur_nf + 4 * L, kernel_size=3, gc=cur_gc, stride=1, bias=True, pad_type='zero',
                            norm_type=norm_type, act_type=act_type, mode='CNA', use_wn=use_wn) for _ in range(nb)]
        LR_conv = B.conv_block(cur_nf + 4 * L, cur_nf + 4 * L, kernel_size=3, norm_type=norm_type, act_type=None,
                               mode=mode,
                               use_wn=use_wn)
        HR_conv0 = B.conv_block(cur_nf + 4 * L, cur_nf, kernel_size=3, norm_type=None, act_type=act_type, use_wn=use_wn)
        if use_sigmoid:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type="sigmoid", use_wn=use_wn)
        else:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type=last_act, use_wn=use_wn)

        self.stack_2 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                    B.upconv_block(cur_nf + 4 * L, cur_nf + 4 * L, act_type=act_type,
                                                   mode=upsample_kernel_mode), HR_conv0, HR_conv1)
        self.L = L
        self.last_act = last_act

    def gen_pe(self, inp):
        if not hasattr(self, 'w_pos'):
            bs, w, h = inp.size(0), inp.size(2), inp.size(3)
            w_pos = torch.linspace(-1, 1, w).reshape(1, w).expand(w, -1)
            h_pos = torch.linspace(-1, 1, h).reshape(h, 1).expand(-1, h)
            periodic_fns = [torch.sin, torch.cos]
            freq_bands = 2. ** torch.arange(self.L) * math.pi
            embed_fns = []
            for freq in freq_bands:
                for p_fn in periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            self.w_pe = torch.stack([embed_fn(w_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(torch.device('cuda'))
            self.h_pe = torch.stack([embed_fn(h_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(torch.device('cuda'))
        return torch.cat((inp, self.w_pe, self.h_pe), dim=1)

    def gen_pe1(self, inp):
        if not hasattr(self, 'w_pos1'):
            bs, w, h = inp.size(0), inp.size(2), inp.size(3)
            w_pos = torch.linspace(-1, 1, w).reshape(1, w).expand(w, -1)
            h_pos = torch.linspace(-1, 1, h).reshape(h, 1).expand(-1, h)
            periodic_fns = [torch.sin, torch.cos]
            freq_bands = 2. ** torch.arange(self.L) * math.pi
            embed_fns = []
            for freq in freq_bands:
                for p_fn in periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            self.w_pe1 = torch.stack([embed_fn(w_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(torch.device('cuda'))
            self.h_pe1 = torch.stack([embed_fn(h_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(torch.device('cuda'))
        return torch.cat((inp, self.w_pe1, self.h_pe1), dim=1)

    def forward(self, img, code, code_1=None, code_2=None):
        x_1 = torch.cat((img, code), dim=1)
        x_1 = self.fea_conv1(x_1)
        x_1 = self.stack_1(self.gen_pe(x_1))
        if self.last_act == "tanh":
            new_result = x_1 + 1
            new_result /= 2.
            x_1 = new_result
        x_2 = torch.cat((x_1, code_1), dim=1)
        x_2 = self.fea_conv2(x_2)
        x_2 = self.stack_2(self.gen_pe1(x_2))
        if self.last_act == "tanh":
            new_result = x_2 + 1
            new_result /= 2.
            x_2 = new_result
        return [x_1, x_2]


class PESkipThreeIMRRDBNet(nn.Module):
    def __init__(self, in_nc, in_code_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu',
                 mode='CNA', upsample_kernel_mode="nearest", use_sigmoid=False, last_act=None, use_wn=False,
                 gcs=[], nfs=[], L=[], device=None):
        super(PESkipThreeIMRRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        self.num_stacks = n_upscale
        cur_gc = gc if len(gcs) < 1 else gcs[0]
        cur_nf = nf if len(nfs) < 1 else nfs[0]

        self.fea_conv1 = B.conv_block(in_nc + in_code_nc, cur_nf, kernel_size=3, norm_type=None, act_type=None,
                                      use_wn=use_wn)
        rb_blocks = [B.RRDB(cur_nf + 4 * L, kernel_size=3, gc=cur_gc, stride=1, bias=True, pad_type='zero',
                            norm_type=norm_type, act_type=act_type, mode='CNA', use_wn=use_wn) for _ in range(nb)]
        LR_conv = B.conv_block(cur_nf + 4 * L, cur_nf + 4 * L, kernel_size=3, norm_type=norm_type, act_type=None,
                               mode=mode, use_wn=use_wn)
        HR_conv0 = B.conv_block(cur_nf + 4 * L, cur_nf, kernel_size=3, norm_type=None, act_type=act_type, use_wn=use_wn)
        if use_sigmoid:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type="sigmoid", use_wn=use_wn)
        else:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type=last_act, use_wn=use_wn)

        self.stack_1 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                   B.upconv_block(cur_nf + 4 * L, cur_nf + 4 * L, act_type=act_type,
                                                  mode=upsample_kernel_mode), HR_conv0, HR_conv1)

        cur_gc = gc if len(gcs) < 1 else gcs[1]
        cur_nf = nf if len(nfs) < 1 else nfs[1]

        self.fea_conv2 = B.conv_block(in_nc + in_code_nc, cur_nf, kernel_size=3, norm_type=None, act_type=None,
                                      use_wn=use_wn)
        rb_blocks = [B.RRDB(cur_nf + 4 * L, kernel_size=3, gc=cur_gc, stride=1, bias=True, pad_type='zero',
                            norm_type=norm_type, act_type=act_type, mode='CNA', use_wn=use_wn) for _ in range(nb)]
        LR_conv = B.conv_block(cur_nf + 4 * L, cur_nf + 4 * L, kernel_size=3, norm_type=norm_type, act_type=None,
                               mode=mode, use_wn=use_wn)

        HR_conv0 = B.conv_block(cur_nf + 4 * L, cur_nf, kernel_size=3, norm_type=None, act_type=act_type, use_wn=use_wn)
        if use_sigmoid:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type="sigmoid", use_wn=use_wn)
        else:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type=last_act, use_wn=use_wn)

        self.stack_2 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                    B.upconv_block(cur_nf + 4 * L, cur_nf + 4 * L, act_type=act_type,
                                                   mode=upsample_kernel_mode),
                                    HR_conv0, HR_conv1)
        cur_gc = gc if len(gcs) < 1 else gcs[2]
        cur_nf = nf if len(nfs) < 1 else nfs[2]

        self.fea_conv = B.conv_block(in_nc + in_code_nc, cur_nf, kernel_size=3, norm_type=None, act_type=None,
                                use_wn=use_wn)
        rb_blocks = [B.RRDB(cur_nf + 4 * L, kernel_size=3, gc=cur_gc, stride=1, bias=True, pad_type='zero',
                            norm_type=norm_type, act_type=act_type, mode='CNA', use_wn=use_wn) for _ in
                     range(nb)]
        LR_conv = B.conv_block(cur_nf + 4 * L, cur_nf + 4 * L, kernel_size=3, norm_type=norm_type, act_type=None,
                               mode=mode, use_wn=use_wn)

        HR_conv0 = B.conv_block(cur_nf + 4 * L, cur_nf, kernel_size=3, norm_type=None, act_type=act_type, use_wn=use_wn)
        if use_sigmoid:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type="sigmoid", use_wn=use_wn)
        else:
            HR_conv1 = B.conv_block(cur_nf, out_nc, kernel_size=3, norm_type=None, act_type=last_act, use_wn=use_wn)
        self.stack_3 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                    B.upconv_block(cur_nf + 4 * L, cur_nf + 4 * L, act_type=act_type,
                                                   mode=upsample_kernel_mode),
                                    HR_conv0, HR_conv1)
        self.L = L
        self.last_act = last_act
        self.device = device

    def gen_pe(self, inp):
        if not hasattr(self, 'w_pos'):
            bs, w, h = inp.size(0), inp.size(2), inp.size(3)
            w_pos = torch.linspace(-1, 1, w).reshape(1, w).expand(w, -1)
            h_pos = torch.linspace(-1, 1, h).reshape(h, 1).expand(-1, h)
            periodic_fns = [torch.sin, torch.cos]
            freq_bands = 2. ** torch.arange(self.L) * math.pi
            embed_fns = []
            for freq in freq_bands:
                for p_fn in periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            self.w_pe = torch.stack([embed_fn(w_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(self.device)
            self.h_pe = torch.stack([embed_fn(h_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(self.device)
        return torch.cat((inp, self.w_pe, self.h_pe), dim = 1)

    def gen_pe1(self, inp):
        if not hasattr(self, 'w_pos1'):
            bs, w, h = inp.size(0), inp.size(2), inp.size(3)
            w_pos = torch.linspace(-1, 1, w).reshape(1, w).expand(w, -1)
            h_pos = torch.linspace(-1, 1, h).reshape(h, 1).expand(-1, h)
            periodic_fns = [torch.sin, torch.cos]
            freq_bands = 2. ** torch.arange(self.L) * math.pi
            embed_fns = []
            for freq in freq_bands:
                for p_fn in periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            self.w_pe1 = torch.stack([embed_fn(w_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(self.device)
            self.h_pe1 = torch.stack([embed_fn(h_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(self.device)
        return torch.cat((inp, self.w_pe1, self.h_pe1), dim = 1)

    def gen_pe2(self, inp):
        if not hasattr(self, 'w_pos2'):
            bs, w, h = inp.size(0), inp.size(2), inp.size(3)
            w_pos = torch.linspace(-1, 1, w).reshape(1, w).expand(w, -1)
            h_pos = torch.linspace(-1, 1, h).reshape(h, 1).expand(-1, h)
            periodic_fns = [torch.sin, torch.cos]
            freq_bands = 2. ** torch.arange(self.L) * math.pi
            embed_fns = []
            for freq in freq_bands:
                for p_fn in periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            self.w_pe2 = torch.stack([embed_fn(w_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(self.device)
            self.h_pe2 = torch.stack([embed_fn(h_pos) for embed_fn in embed_fns],
                                    dim=0).unsqueeze(0).expand(bs, -1, -1, -1).to(self.device)
        return torch.cat((inp, self.w_pe2, self.h_pe2), dim = 1)

    def forward(self, img, code, code_1=None, code_2=None):
        x_1 = torch.cat((img, code), dim=1)
        x_1 = self.fea_conv1(x_1)
        x_1 = self.stack_1(self.gen_pe(x_1))
        if self.last_act == "tanh":
            new_result = x_1 + 1
            new_result /= 2.
            x_1 = new_result
        x_2 = torch.cat((x_1, code_1), dim=1)
        x_2 = self.fea_conv2(x_2)
        out_2 = self.stack_2(self.gen_pe1(x_2))
        if self.last_act == "tanh":
            new_result = out_2 + 1
            new_result /= 2.
            x_2 = new_result
        x_3 = torch.cat((x_2, code_2), dim=1)
        x_3 = self.fea_conv(x_3)
        x_3 = self.stack_3(self.gen_pe2(x_3))
        if self.last_act == "tanh":
            new_result = x_3 + 1
            new_result /= 2.
            x_3 = new_result
        return [x_1, x_2, x_3]
