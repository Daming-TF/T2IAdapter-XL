#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export INPUT1='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2'
export INPUT2='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2/14_v2.jpg'
export OUTPUT='/home/mingjiahui/data/T2IAdapter/additional_scale_test/0_debug'
# 21_v2-modify7_0_cond.jpg  # 21_v2_modify6_0_cond.jpg  # 21_v2_modify5_0_cond.jpg  # 21_v2-modify2_0_cond.jpg
# 21_v2-modify_cond.jpg   # 21_v2_modify3_0_cond.jpg   # 21_v2-modify4_0_cond.jpg
# export COND='./data/21_v2_modify5_0_cond.jpg'
#export COND='/home/mingjiahui/data/T2IAdapter/offical/sketch/14_v2.png'
export COND='/home/mingjiahui/data/T2IAdapter/offical/sketch/'
python test_df2_lineart_reconstruction.py \
      --input=$INPUT1 \
      --output=$2 \
      --batch_size=4 \
      --seed=42 \
      --resolution=1024 \
      --steps=50 \
      --scale=8 \
#      --additional_scale=$3 \
#      --color_inversion=True \
#      --inversion_ratio=$4 \
#      --cond=$COND \
      # --prompt='Two guinea pigs with ribbons stuck together, Christmas hat, in front of a Christmas tree' \
      # 'Two guinea pigs，side by side, Christmas hat, in front of a Christmas tree'


