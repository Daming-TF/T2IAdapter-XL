import torch


def save_sdxl_adapter_checkpoint(output_file, sdxl_adapter, epochs, steps):
    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            # if save_dtype is not None:
            #     v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    # Convert the UNet model
    update_sd("model.sdxl_adapter.", sdxl_adapter.state_dict())

    # Put together new checkpoint
    key_count = len(state_dict.keys())
    new_ckpt = {"state_dict": state_dict}

    # # epoch and global_step are sometimes not int
    # if ckpt_info is not None:
    #     epochs += ckpt_info[0]
    #     steps += ckpt_info[1]

    new_ckpt["epoch"] = epochs
    new_ckpt["global_step"] = steps

    torch.save(new_ckpt, output_file)

    return key_count


def load_sdxl_adapter_chaeckpoit(ckpt_path, ):
    print(f'Loading the adapter pretrain ckpt...... ==>\t{ckpt_path}')
    my_dict = torch.load(ckpt_path)

    def update_sd(prefix, sd):
        status = {}
        for k, v in sd.items():
            # if save_dtype is not None:
            #     v = v.detach().clone().to("cpu").to(save_dtype)
            status[k.replace(prefix, '')] = v
        return status

    epoch = my_dict['epoch']
    global_step = my_dict['global_step']
    state_dict = my_dict['state_dict']

    state_dict = update_sd("model.sdxl_adapter.", state_dict)
    for param_name, param_tensor in state_dict.items():
        state_dict[param_name] = param_tensor.to(torch.float16)
    return epoch, global_step, state_dict


if __name__ == '__main__':
    path = r'/mnt/nfs/file_server/public/mingjiahui/experiments/T2IAdapter-sdxl/test3-160K/chenckpoint/e1-i10000.ckpt'
    _, _, my_dict = load_sdxl_adapter_chaeckpoit(path)
    for param_name, param_tensor in my_dict.items():
        print(f"Parameter: {param_name}, Data Type: {param_tensor.dtype}")
        exit(0)
