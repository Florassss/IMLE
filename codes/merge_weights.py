import torch


def load_partial_network(load_path1, load_path2, save_path):
    dict1 = torch.load(load_path1)
    dict2 = torch.load(load_path2)
    result = dict()
    for k, v in dict1.items():
        if "stack" in k:
            new_name1 = k
            new_name2 = k[:6] + "2" + k[7:]
        elif "ext_conv" in k:
            new_name1 = k[:8] + "1" + k[8:]
            new_name2 = k[:8] + "2" + k[8:]
        result[new_name1] = v
        result[new_name2] = dict2[k]
        # print(new_name1, new_name2)
    torch.save(result, save_path)


def load_partial_network_skip(load_path1, load_path2, save_path):
    dict1 = torch.load(load_path1)
    dict2 = torch.load(load_path2)
    result = dict()
    for k, v in dict1.items():
        if "stack" in k:
            if "stack_1.7" in k:
                new_name1 = "stack_1_8.0" + k[9:]
            else:
                new_name1 = k
            new_name2 = k[:6] + "2" + k[7:]
        elif "ext_conv" in k:
            new_name1 = k[:8] + "1" + k[8:]
            new_name2 = k[:8] + "2" + k[8:]
        result[new_name1] = v
        # if "stack_2.0.weight_g" in new_name2:
        #     weight_shape = dict2[k].shape
        #     print(new_name2, weight_shape)
        #     print(torch.mean(dict2[k]), torch.std(dict2[k]))
        if "stack_2.0.weight_v" in new_name2:
            weight_shape = dict2[k].shape
            print(new_name2, weight_shape)
            print(torch.mean(dict2[k]), torch.std(dict2[k]))
            concate_weights = torch.normal(mean=torch.zeros((weight_shape[0], 128, weight_shape[2],
                                                             weight_shape[3])), std=0.08)
            combined_weight = torch.cat((dict2[k], concate_weights), dim=1)
            # result[new_name2] = combined_weight
        else:
            result[new_name2] = dict2[k]
        # print(new_name1, new_name2)
    torch.save(result, save_path)


def load_partial_network_skip3(load_path1, load_path2, save_path):
    dict1 = torch.load(load_path1)
    dict2 = torch.load(load_path2)
    result = dict()
    for k, v in dict1.items():
        # if "stack" in k:
        #     if "stack_2.7" in k:
        #         new_name1 = "stack_2_8.0" + k[9:]
        #     else:
        #         new_name1 = k
        # else:
        new_name1 = k
            # print(k)
        result[new_name1] = v
    for k, v in dict2.items():
        if "stack" in k:
            new_name2 = k[:6] + "3" + k[7:]
        else:
            new_name2 = k
        # if "stack_3.0.weight_v" in new_name2:
        #     weight_shape = dict2[k].shape
        #     print(new_name2, weight_shape)
        #     print(torch.mean(dict2[k]), torch.std(dict2[k]))
        #     concate_weights = torch.normal(mean=torch.zeros((weight_shape[0], 64, weight_shape[2],
        #                                                      weight_shape[3])), std=0.09)
        #     combined_weight = torch.cat((dict2[k], concate_weights), dim=1)
        #     result[new_name2] = combined_weight
        # else:
        print(new_name2)
        result[new_name2] = dict2[k]
    torch.save(result, save_path)


def save_partial_network(load_path, save_path):
    dict1 = torch.load(load_path)
    result = dict()
    for k, v in dict1.items():
        if "stack_3.0.weight_v" in k:
            result[k] = v[:, :8, :, :]
        else:
            result[k] = v
    torch.save(result, save_path)


def save_third_stack(load_path, save_path):
    dict1 = torch.load(load_path)
    result = dict()
    for k, v in dict1.items():
        if "stack_3" in k:
            new_name = k[:6] + "1" + k[7:]
            result[new_name] = v
        else:
            result[k] = v
    torch.save(result, save_path)


def adjust_partial_network(load_path, save_path):
    dict1 = torch.load(load_path)
    result = dict()
    for k, v in dict1.items():
        new_name = k.replace("model", "stack_1")
        print(new_name)
        result[new_name] = v
    torch.save(result, save_path)


if __name__ == '__main__':
    # path1 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_17_x2_Butterfly/models/1000000_G.pth"
    # path2 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_5_x2_Butterfly/models/540000_G.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_25_x4_init.pth"
    # load_partial_network(path1, path2, save_p)

    # path1 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_17_x2_Butterfly/models/1000000_G.pth"
    # path2 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_5_x2_Butterfly/models/540000_G.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_29_x4_init_1.pth"
    # load_partial_network_skip(path1, path2, save_p)

    # path1 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_29_x4_Butterfly/models/220000_G.pth"
    # path2 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_2_x2_Butterfly/models/120000_G.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_13_x8_init.pth"
    # load_partial_network_skip3(path1, path2, save_p)

    # path1 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_37_x4_Bird/models/295000_G.pth"
    # path2 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_21_x2_Bird/models/135000_G.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_18_x8_init.pth"
    # load_partial_network_skip3(path1, path2, save_p)
    #
    # path1 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_38_x4_Straberry/models/375000_G.pth"
    # path2 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_22_x2_Strawberry/models/135000_G.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_19_x8_init.pth"
    # load_partial_network_skip3(path1, path2, save_p)

    # path1 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_39_x4_Cicada/models/285000_G.pth"
    # path2 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_23_x2_Cicada/models/135000_G.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_20_x8_init.pth"
    # load_partial_network_skip3(path1, path2, save_p)

    # src_p = "/home/nio/SRIM/experiments/pretrained_models/ms_13_x8_init.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_14_x8_init.pth"
    # save_partial_network(src_p, save_p)

    # src_p = "/home/s5peng/pretrained/ms_13_x8_init.pth"
    # save_p = "/home/s5peng/pretrained/ms_14_x8_init.pth"
    # save_partial_network(src_p, save_p)

    # path1 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_42_x4_Butterfly/models/230000_G.pth"
    # path2 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_29_x2_Butterfly/models/120000_G.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_23_x8_init.pth"
    # load_partial_network_skip3(path1, path2, save_p)

    # path1 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_48_x4_Lamp/models/140000_G.pth"
    # path2 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_28_x2_Lamp/models/100000_G.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_24_x8_init.pth"
    # load_partial_network_skip3(path1, path2, save_p)

    # src_p = "/home/nio/SRIM/experiments/RRDB_IM_RRDB_2_x2_Butterfly/models/500000_G.pth"
    # save_p = "/home/nio/SRIM/experiments/pretrained_models/im_0_x4_init.pth"
    # adjust_partial_network(src_p, save_p)

    # path1 = "/home/nio/pretrained/ms_18_x8.pth"
    # save_p = "/home/nio/pretrained/ms_18_x8_third.pth"
    # save_third_stack(path1, save_p)

    path1 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_33_x4_Bird/models/240000_G.pth"
    path2 = "/home/nio/SRIM/experiments/RRDB_MS_RRDB_21_x2_Bird/models/135000_G.pth"
    save_p = "/home/nio/SRIM/experiments/pretrained_models/ms_25_x8_init.pth"
    load_partial_network_skip3(path1, path2, save_p)
