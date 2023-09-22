import os
import json
from tqdm import tqdm
# targe_dir = r'/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/laion_12m_aesthetics'
targe_dir = r'/mnt/nfs/file_server/public/lipengxiang/improved_aesthetics_6plus_out/'
# depth_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/laion_12m_aesthetics/lineart_align_controlnet_reso512'
depth_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/lineart_align_controlnet/'
inference_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2/'
save_path1 = r'./data/train_data_v1-1M-reso1024-lineart.json'
save_path2 = r'./data/overfit_data_v1-1M-reso1024-lineart.json'
# os.makedirs(os.path.dirname(save_path1), exist_ok=True)

# # V1
# data={}
# print('get the dirs')
# img_dirs:[list] = [os.path.join(targe_dir, name) for name in tqdm(os.listdir(targe_dir))
#                    if os.path.isdir(os.path.join(targe_dir, name)) and int(name)<200]
# for img_dir in tqdm(img_dirs):
#     img_paths:[list] = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith('.jpg')]
#     print(f'{img_dir} Img num:{len(img_paths)}')
#     for img_path in img_paths:
#         if img_path not in data.keys():
#             data[img_path] = {}
#         # get caption
#         data[img_path]['txt'] = img_path.replace('.jpg', '.txt')
#         # get depth
#         model_type = "midas_v21_small_256"
#         depth_name = os.path.splitext(os.path.basename(img_path))[0] + '-' + model_type + '.png'
#         dir_name = os.path.basename(os.path.dirname(img_path))
#         depth_path = os.path.join(depth_dir, dir_name, depth_name)
#         print(depth_path)
#         data[img_path]['depth'] = depth_path
#
# debug_dict = {key: data[key] for key in list(data.keys())[:4]}
# with open(save_path2, 'w')as file:
#     json.dump(debug_dict, file)
#
# print('writing in json......')
# with open(save_path1, 'w')as file:
#     json.dump(data, file)


def process():
    # V2
    data = {}
    print('get the dirs')
    # img_dirs: [list] = [os.path.join(targe_dir, name) for name in tqdm(os.listdir(targe_dir))
    #                    if os.path.isdir(os.path.join(targe_dir, name)) and int(name) < 123]     # 123 200
    img_dirs: [list] = [os.path.join(targe_dir, name) for name in tqdm(os.listdir(targe_dir))
                        if os.path.isdir(os.path.join(targe_dir, name))]  # 123 200

    loss_num = 0
    for img_dir in tqdm(img_dirs):
        img_paths:[list] = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith('.jpg')]
        for img_path in img_paths:
            # get depth
            # model_type = "dpt-hybrid-midas"
            # depth_name = os.path.splitext(os.path.basename(img_path))[0] + '-' + model_type + '.png'
            depth_name = os.path.splitext(os.path.basename(img_path))[0] + '.png'
            dir_name = os.path.basename(os.path.dirname(img_path))
            depth_path = os.path.join(depth_dir, dir_name, depth_name)
            txt_path = img_path.replace('.jpg', '.txt')

            if not os.path.exists(depth_path) or not os.path.exists(txt_path):
                print(f'This depth is loss:\t{depth_path}\t num:{loss_num}')
                loss_num += 1
                continue
            if img_path not in data.keys():
                data[img_path] = {}

            data[img_path]['txt'] = txt_path
            data[img_path]['depth'] = depth_path
    print(rf'loss depth num is:{loss_num}')

    print('writing in json......')
    with open(save_path1, 'w')as file:
        json.dump(data, file)
        print(f'ToTal num:{len(data)} ==> {save_path1}')

    # debug_dict = {key: data[key] for key in list(data.keys())[:4]}
    image_paths = [os.path.join(inference_dir, name) for name in os.listdir(inference_dir) if name.endswith('jpg')]
    res_dict = {}
    for img_path in tqdm(image_paths):
        print(img_path)
        txt_path = img_path.replace('.jpg', '.txt')
        dep_path = img_path.replace('.jpg', '-orisize.png')
        if img_path not in res_dict.keys():
            res_dict[img_path] = {}
        res_dict[img_path]['txt'] = txt_path
        res_dict[img_path]['depth'] = dep_path
    with open(save_path2, 'w')as file:
        json.dump(res_dict, file)
        print(f'ToTal num:{len(res_dict)} ==> {save_path2}')


if __name__ == '__main__':
    process()


