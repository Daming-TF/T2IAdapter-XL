#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export INPUT1='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2'
export INPUT2='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2/21_v2.jpg'
export OUTPUT='/home/mingjiahui/data/T2IAdapter/additional_scale_test/0_debug'
# 21_v2-modify7_0_cond.jpg  # 21_v2_modify6_0_cond.jpg  # 21_v2_modify5_0_cond.jpg  # 21_v2-modify2_0_cond.jpg
# 21_v2-modify_cond.jpg   # 21_v2_modify3_0_cond.jpg   # 21_v2-modify4_0_cond.jpg
export COND='./data/21_v2_modify5_0_cond.jpg'
python test.py \
      --config='config/inference/'


