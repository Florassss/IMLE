import torch
import sys
import numpy as np
from dci import DCI


def print_without_newline(s):
   sys.stdout.write(s)
   sys.stdout.flush()


def get_code_for_data_one_lpips(model, data, opt):
    options = opt['train']

    code_nc = int(opt['network_G']['in_code_nc'])
    pull_num_sample_per_img = int(options['num_code_per_img'])

    pull_gen_img = data['LR']
    real_gen_img = data['HR']
    sample_perturbation_magnitude = float(options['sample_perturbation_magnitude'])

    forward_bs = 20

    print("Generating Pull Samples")
    data_length = pull_gen_img.shape[0]

    pull_gen_code_0 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2], pull_gen_img.shape[3])
    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding first stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        pull_gen_code_pool_0 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2], pull_gen_img.shape[3])
        losses = torch.zeros(pull_num_sample_per_img)
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target}
            start = i
            end = i + forward_bs
            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end]])
            cur_loss = model.get_loss(0)
            losses[start:end] = torch.squeeze(cur_loss)
        pull_gen_code_0[sample_index, :] = pull_gen_code_pool_0[int(torch.argmin(losses)), :]

    print('\rFinding first stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    pull_gen_code_0 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2],
                                                                   pull_gen_img.shape[3])

    return [pull_gen_code_0]


def get_code_for_data_two_lpips(model, data, opt):
    options = opt['train']

    code_nc = int(opt['network_G']['in_code_nc'])
    pull_num_sample_per_img = int(options['num_code_per_img'])

    pull_gen_img = data['LR']
    d1_gen_img = data['D1']
    d2_gen_img = data['D2']
    real_gen_img = data['HR']
    sample_perturbation_magnitude = float(options['sample_perturbation_magnitude'])

    forward_bs = 20

    print("Generating Pull Samples")
    data_length = pull_gen_img.shape[0]

    pull_gen_code_0 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2], pull_gen_img.shape[3])
    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding first stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        pull_gen_code_pool_0 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2], pull_gen_img.shape[3])
        pull_gen_code_pool_1 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 2,
                                           pull_gen_img.shape[3] * 2)
        pull_gen_code_pool_2 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 4,
                                           pull_gen_img.shape[3] * 4)
        losses = torch.zeros(pull_num_sample_per_img)
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d2 = d2_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1, 'D2': pull_d2}
            start = i
            end = i + forward_bs
            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end],
                                            pull_gen_code_pool_2[start:end]])
            cur_loss = model.get_loss(0)
            losses[start:end] = torch.squeeze(cur_loss)
        pull_gen_code_0[sample_index, :] = pull_gen_code_pool_0[int(torch.argmin(losses)), :]

    print('\rFinding first stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    # ============ ADD POINT ==================
    pull_gen_code_1 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2] * 2, pull_gen_img.shape[3] * 2)

    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding second stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        # ============ ADD POINT ==================
        pull_gen_code_pool_0 = pull_gen_code_0[sample_index].expand(pull_num_sample_per_img, -1, -1, -1)
        pull_gen_code_pool_1 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 2,
                                           pull_gen_img.shape[3] * 2)
        pull_gen_code_pool_2 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 4,
                                           pull_gen_img.shape[3] * 4)
        losses = torch.zeros(pull_num_sample_per_img)
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d2 = d2_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1, 'D2': pull_d2}
            start = i
            end = i + forward_bs

            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end],
                                            pull_gen_code_pool_2[start:end]])
            cur_loss = model.get_loss(1)
            losses[start:end] = torch.squeeze(cur_loss)
        pull_gen_code_1[sample_index, :] = pull_gen_code_pool_1[int(torch.argmin(losses, dim=0)), :]

    print('\rFinding second stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    pull_gen_code_0 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2],
                                                                   pull_gen_img.shape[3])
    pull_gen_code_1 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2] * 2,
                                                                   pull_gen_img.shape[3] * 2)

    return [pull_gen_code_0, pull_gen_code_1]


def get_code_for_data_three_lpips(model, data, opt):
    options = opt['train']

    code_nc = int(opt['network_G']['in_code_nc'])
    pull_num_sample_per_img = int(options['num_code_per_img'])

    pull_gen_img = data['LR']
    d1_gen_img = data['D1']
    d2_gen_img = data['D2']
    real_gen_img = data['HR']
    sample_perturbation_magnitude = float(options['sample_perturbation_magnitude'])

    forward_bs = 20

    print("Generating Pull Samples")
    data_length = pull_gen_img.shape[0]

    pull_gen_code_0 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2], pull_gen_img.shape[3])
    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding first stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        pull_gen_code_pool_0 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2], pull_gen_img.shape[3])
        pull_gen_code_pool_1 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 2,
                                           pull_gen_img.shape[3] * 2)
        pull_gen_code_pool_2 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 4,
                                           pull_gen_img.shape[3] * 4)
        losses = torch.zeros(pull_num_sample_per_img)
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d2 = d2_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1, 'D2': pull_d2}
            start = i
            end = i + forward_bs
            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end],
                                            pull_gen_code_pool_2[start:end]])
            cur_loss = model.get_loss(0)
            losses[start:end] = torch.squeeze(cur_loss)
        pull_gen_code_0[sample_index, :] = pull_gen_code_pool_0[int(torch.argmin(losses)), :]

    print('\rFinding first stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    # ============ ADD POINT ==================
    pull_gen_code_1 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2] * 2, pull_gen_img.shape[3] * 2)

    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding second stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        # ============ ADD POINT ==================
        pull_gen_code_pool_0 = pull_gen_code_0[sample_index].expand(pull_num_sample_per_img, -1, -1, -1)
        pull_gen_code_pool_1 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 2,
                                           pull_gen_img.shape[3] * 2)
        pull_gen_code_pool_2 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 4,
                                           pull_gen_img.shape[3] * 4)
        losses = torch.zeros(pull_num_sample_per_img)
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d2 = d2_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1, 'D2': pull_d2}
            start = i
            end = i + forward_bs

            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end],
                                            pull_gen_code_pool_2[start:end]])
            cur_loss = model.get_loss(1)
            losses[start:end] = torch.squeeze(cur_loss)
        pull_gen_code_1[sample_index, :] = pull_gen_code_pool_1[int(torch.argmin(losses, dim=0)), :]

    print('\rFinding second stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    # ============ ADD POINT ==================
    pull_gen_code_2 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2] * 4, pull_gen_img.shape[3] * 4)

    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding third stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        # ============ ADD POINT ==================
        pull_gen_code_pool_0 = pull_gen_code_0[sample_index].expand(pull_num_sample_per_img, -1, -1, -1)
        pull_gen_code_pool_1 = pull_gen_code_1[sample_index].expand(pull_num_sample_per_img, -1, -1, -1)
        pull_gen_code_pool_2 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 4,
                                           pull_gen_img.shape[3] * 4)
        losses = torch.zeros(pull_num_sample_per_img)
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d2 = d2_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1, 'D2': pull_d2}
            start = i
            end = i + forward_bs

            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end],
                                            pull_gen_code_pool_2[start:end]])
            cur_loss = model.get_loss(2)
            losses[start:end] = torch.squeeze(cur_loss)
        pull_gen_code_2[sample_index, :] = pull_gen_code_pool_2[int(torch.argmin(losses, dim=0)), :]

    print('\rFinding third stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    pull_gen_code_0 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2],
                                                                   pull_gen_img.shape[3])
    pull_gen_code_1 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2] * 2,
                                                                   pull_gen_img.shape[3] * 2)
    pull_gen_code_2 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2] * 4,
                                                                   pull_gen_img.shape[3] * 4)

    return [pull_gen_code_0, pull_gen_code_1, pull_gen_code_2]
