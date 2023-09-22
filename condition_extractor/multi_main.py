"""
TODO：V1和V2主要区别在于读取图片路径在多进程里面还是主进程里，对于数据量较大推荐使用V2，但是V1能够更使每个进程处理的图片更均匀分配
"""
import multiprocessing
import os
import sys
from tqdm import tqdm
import argparse
import logging

current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))


def main(args, process):
    multiprocessing.set_start_method('spawn')
    logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # prepare img path
    def _get_img_path(input_path):
        res = []

        # TODO: debug
        # img_dirs = [os.path.join(input_path, name) for name in os.listdir(input_path)
        #             if os.path.isdir(os.path.join(input_path, name)) and int(name) < 1]

        img_dirs = [os.path.join(input_path, name) for name in os.listdir(input_path)
                    if os.path.isdir(os.path.join(input_path, name))]

        for img_dir in tqdm(img_dirs):
            # if '00054' in img_dir:
            #     print(len(res))
            img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith('.jpg')]
            res += img_paths
        return res
    image_list = _get_img_path(args.input_path)
    num = len(image_list) // args.num_processes

    # get gpu ids
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_ids = [int(num) for num in cuda_visible_devices.split(',')[:-1]]
    assert len(gpu_ids) == args.num_processes

    # prepare multi processes
    processes = []
    for i in range(args.num_processes):
        gpu_id = gpu_ids[i]
        images_per_process = image_list[i * num:] \
            if i == args.num_processes - 1 else image_list[i * num: (i + 1) * num]
        print(f'{i}/{gpu_id}: {i * num} ~ {i * num + len(images_per_process)}')
        p = multiprocessing.Process(target=process, args=(images_per_process, args, i))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        default=r'/mnt/nfs/file_server/public/lipengxiang/improved_aesthetics_6plus_out/',
        type=str,
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--cond_type',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=True,
    )
    args = parser.parse_args()

    from condition_extractor import process_depth, process_lineart
    process_dict = {
        'depth': process_depth,
        'lineart': process_lineart,
    }
    process = process_dict[args.cond_type]

    main(args, process)
