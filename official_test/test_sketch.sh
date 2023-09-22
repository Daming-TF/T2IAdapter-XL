#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export MODEL_PATH='/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/'
export INPUT1='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2'
export INPUT2='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2/14_v2.jpg'
export OUTPUT='/home/mingjiahui/data/T2IAdapter/offical/test1'
# 21_v2-modify7_0_cond.jpg  # 21_v2_modify6_0_cond.jpg  # 21_v2_modify5_0_cond.jpg  # 21_v2-modify2_0_cond.jpg
# 21_v2-modify_cond.jpg   # 21_v2_modify3_0_cond.jpg   # 21_v2-modify4_0_cond.jpg
#export COND='./data/21_v2_modify5_0_cond.jpg'
python official_test/test-official_sketch-test_lineart.py \
      --model_id=$MODEL_PATH \
      --config='configs/inference/Adapter-XL-sketch.yaml' \
      --in_type=image \
      --scale=8.0 \
      --step=50 \
      --seed=42 \
      --input=$INPUT1 \
      --output=$OUTPUT \
      --neg_prompt='' \
#      --prompt='Ancient style, a tree in the middle of a forest, with a temple in the background'



