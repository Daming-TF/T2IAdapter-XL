export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,
export INPUT='/mnt/nfs/file_server/public/lipengxiang/improved_aesthetics_6plus_out/'
#export INPUT='/mnt/nfs/file_server/public/wangrui/sd_finetune_lion400/laion_12m_aesthetics'   # '/mnt/nfs/file_server/public/lipengxiang/improved_aesthetics_6plus_out/'
export OUTPUT='/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/lineart_align_controlnet/'
#export OUTPUT='/mnt/nfs/file_server/public/mingjiahui/data/laion_12m_aesthetics/lineart_align_controlnet_reso512'
/home/mingjiahui/anaconda3/envs/T2I/bin/python condition_extractor/multi_main_v2.py \
      --input_path=$INPUT \
      --output_path=$OUTPUT \
      --batch_size=30 \
      --num_processes=7 \
      --cond_type=lineart \
      --resolution=1024 \
      --start_index=5200 \
#      --data_num=123
