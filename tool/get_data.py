import os
import json
from tqdm import tqdm
targe_dir = r'/mnt/nfs/file_server/public/lipengxiang/improved_aesthetics_6plus_out'
depth_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/depth_align_controlnet'
inference_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test/'
save_path1 = r'./data/train_data_v1_1.json'
save_path2 = r'./data/overfit_data_v1_1.json'
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

# V2
data = {}
print('get the dirs')
img_dirs: [list] = [os.path.join(targe_dir, name) for name in tqdm(os.listdir(targe_dir))
                   if os.path.isdir(os.path.join(targe_dir, name)) and int(name)<200]

loss_num = 0
for img_dir in tqdm(img_dirs):
    img_paths:[list] = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith('.jpg')]
    for img_path in img_paths:
        # get depth
        model_type = "dpt-hybrid-midas"
        depth_name = os.path.splitext(os.path.basename(img_path))[0] + '-' + model_type + '.png'
        dir_name = os.path.basename(os.path.dirname(img_path))
        depth_path = os.path.join(depth_dir, dir_name, depth_name)

        if not os.path.exists(depth_path):
            print(depth_path)
            loss_num += 1
            continue
        if img_path not in data.keys():
            data[img_path] = {}

        data[img_path]['txt'] = img_path.replace('.jpg', '.txt')
        data[img_path]['depth'] = depth_path
print(rf'loss depth num is:{loss_num}')

print('writing in json......')
with open(save_path1, 'w')as file:
    json.dump(data, file)
    print(f'ToTal num:{len(data)} ==> {save_path1}')

debug_dict = {key: data[key] for key in list(data.keys())[:4]}
image_paths = [os.path.join(inference_dir, name) for name in os.listdir(inference_dir) if name.endswith('jpg')]
for img_path in tqdm(image_paths):
    print(img_path)
    txt_path = img_path.replace('.jpg', '.txt')
    dep_path = img_path.replace('.jpg', '-dpt-hybrid-midas.png')
    if img_path not in debug_dict.keys():
        debug_dict[img_path] = {}
    debug_dict[img_path]['txt'] = txt_path
    debug_dict[img_path]['depth'] = dep_path
with open(save_path2, 'w')as file:
    json.dump(debug_dict, file)
    print(f'ToTal num:{len(debug_dict)} ==> {save_path2}')



