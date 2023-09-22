#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export INPUT1='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2'
export INPUT2='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2/21_v2.jpg'
export OUTPUT='/home/mingjiahui/data/T2IAdapter/additional_scale_test/0_debug'
# 21_v2-modify7_0_cond.jpg  # 21_v2_modify6_0_cond.jpg  # 21_v2_modify5_0_cond.jpg  # 21_v2-modify2_0_cond.jpg
# 21_v2-modify_cond.jpg   # 21_v2_modify3_0_cond.jpg   # 21_v2-modify4_0_cond.jpg
export COND='./data/21_v2_modify5_0_cond.jpg'
python test_df2_lineart.py \
      --input=$INPUT2 \
      --output=$3 \
      --batch_size=1 \
      --seed=42 \
      --resolution=1024 \
      --steps=50 \
      --scale=7.5 \
      --additional_scale=$2 \
#      --cond=$COND \
#     --prompt='Two guinea pigs with ribbons stuck together, Christmas hat, in front of a Christmas tree' \
      # 'Two guinea pigs，side by side, Christmas hat, in front of a Christmas tree'


