# ToDo: 对于像调用wenbui批量跑图时有大量结果，该脚本设计的目的就是为了更好的管理实验结果
#  e.g. 在controlnet和t2iadapter定量分析实验时，设定了control mode∈(0, 1, 2), control scale∈(0, 1.2, 0.2), 两组lora
#       可以提前把变量名与文件名设定为键值对以键值方式设定需要concat图片的xy值分别是啥
"""
input:
    --xy:       每张图生成一个xy大图，改参数控制xy分别表示什么变量
    --lora:     若lora非xy变量则给定固定值，否则不用输入default None/ Option:[0,1]
    --model:    若model非xy变量则给定固定值，否则不用输入default None/ Option:[0,1]
    --mode:    若model非xy变量则给定固定值，否则不用输入default None/ Option:[0,1,2]
    --scale:    若model非xy变量则给定固定值，否则不用输入default None/ Option:[0,0.2,0.4,0.6,0.8,1.0]
output:
    1.每张图片输入对应的xy大图
    2.每次实验将保存在--save_dir下的 ’xy-<x>-<y>-<fix variable>‘文件夹
    3.在2提及路径下保存所有实验参数的json
"""
import json
import numpy as np
import os
import copy
from argparse import ArgumentParser
import cv2
from tqdm import tqdm


def set_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--xy', type=str, nargs='+', required=True,
    )
    parser.add_argument(
        '--image_dir', type=str, default='/home/mingjiahui/data/test_data/sketch_10'
    )
    parser.add_argument(
        '--input_dir', type=str, default='/home/mingjiahui/data/result/sketch'
    )
    parser.add_argument('--lora', type=str, default=None, choices=['0', '1'])
    parser.add_argument('--model', type=str, default=None, choices=['0', '1'])
    parser.add_argument('--mode', type=str, default=None, choices=['0', '1', '2'])
    parser.add_argument('--scale', type=str, default=None, choices=['0', '0.20', '0.40', '0.60', '0.80', ' 1.00'])
    parser.add_argument('--image_id', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', required=True, type=str)
    return parser.parse_args()


def check_v(fix_v, args, v_dict):
    [x, y] = args.xy
    fix_v_state_dict = {}
    if getattr(args, x) is not None or getattr(args, y) is not None:
        print(f'x, y is x/y,can not set fix value in agrs')
        exit(0)

    print('CHecking fix value')
    for key in fix_v:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is None:
                print(f'>> --{key} << is a fix variable,but it was defaulted None')
                exit()
            else:
                print(f'{key}:\t{getattr(args, key)}')
                fix_v_state_dict[key] = v_dict[key][getattr(args, key)] if key != 'scale' else getattr(args, key)
                continue
        else:
            print(f'error args: >>{key}<<')

    return fix_v_state_dict


def main(args):
    # init
    lora_dict = {
        '0': '<lyco:8Bit_Scenes:0.6><lyco:8Bit_Objects:0.3>', # '8Bit_Scenes_0.6-8Bit_Objects_0.3',
        '1': '<lyco:Paper_Cutout:1.0>' # 'Paper_Cutout_1.0',
    }
    model_dict = {
        '0': 'control_v11p_sd15_scribble-scribble_pidinet',
        '1': 't2iadapter_sketch_sd15v2-t2ia_sketch_pidi',
    }
    control_mode = {
        '0': '0-balanced',
        # '1': '1-prompt_is_important',
        # '2': '2-controlnet_is_import',
    }
    control_scale = np.arange(0.6, 1.1, 0.1).tolist()
    control_scale = ['{:.2f}'.format(round(num, 2))for num in control_scale]
    print(control_scale)
    v_dict = {
        'lora': lora_dict,
        'model': model_dict,
        'mode': control_mode,
        'scale': control_scale,
    }

    [x, y] = args.xy
    x_group = v_dict[x]
    y_group = v_dict[y]
    fix_v = copy.deepcopy(v_dict)
    fix_v.pop(x)
    fix_v.pop(y)
    fix_v = list(fix_v.keys())

    fix_v_state_dict = check_v(fix_v, args, v_dict)
    os.makedirs(args.save_dir, exist_ok=True)

    # run
    image_ids = [name.split('.')[0] for name in os.listdir(args.image_dir)]
    print(image_ids)
    # exit(0)

    for image_id in tqdm(image_ids):
        result = None
        for x_value in x_group:
            img_vconcat = None
            for y_value in y_group:
                img_search_dir = args.input_dir

                # get image dir
                for key in v_dict.keys():
                    if x == key:
                        if isinstance(x_group, list):
                            img_search_dir = os.path.join(img_search_dir, x_value)
                        elif isinstance(x_group, dict):
                            img_search_dir = os.path.join(img_search_dir, x_group[x_value])
                        else:
                            print('some error happens')
                            exit(0)
                    elif y == key:
                        if isinstance(y_group, list):
                            img_search_dir = os.path.join(img_search_dir, y_value)
                        elif isinstance(y_group, dict):
                            img_search_dir = os.path.join(img_search_dir, y_group[y_value])
                        else:
                            print('some error happens')
                            exit(0)
                    else:
                        img_search_dir = os.path.join(img_search_dir, v_dict[key][getattr(args, key)]) \
                            if key != 'scale'else os.path.join(img_search_dir, getattr(args, key))

                if not os.path.exists(img_search_dir):
                    print(f'ERROR is happened. image dir is not exists ==> **{img_search_dir}')
                    exit(1)

                save_name = f"{image_id}-{args.seed}.jpg"
                img = cv2.imread(os.path.join(img_search_dir, save_name))
                print(f'loading ==> {os.path.join(img_search_dir, save_name)}')
                img_vconcat = img if img_vconcat is None else cv2.vconcat([img_vconcat, img])

            result = img_vconcat if result is None else cv2.hconcat([result, img_vconcat])

        save_name = f'xy-{x}-{y}'
        for key in fix_v:
            save_key = f'{key}_{v_dict[key][getattr(args, key)]}' if key != 'scale' else f'{key}_{getattr(args, key)}'
            save_name += f'-{save_key}'
        save_dir = os.path.join(args.save_dir, save_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_id+'.jpg')
        print(f'image save to ==> {save_path}')
        cv2.imwrite(save_path, result)

        # save_json
        print('saving the args states')
        state_dict = {
            'x': {
                'name': x,
                'range': x_group,
            },
            'y': {
                'name': y,
                'range': y_group,
            }
        }
        state_dict.update(fix_v_state_dict)
        json_path = os.path.join(save_dir, 'state_dict.json')
        with open(json_path, 'w')as f:
            json.dump(state_dict, f, indent=4)


if __name__ == '__main__':
    args = set_args()
    main(args)
