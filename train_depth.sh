export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MODEL_PATH='/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/'   # stabilityai/stable-diffusion-xl-base-1.0
export OUTPUT_DIR='/mnt/nfs/file_server/public/mingjiahui/experiments/T2IAdapter-sdxl/test4-160K'
/home/mingjiahui/anaconda3/envs/T2I/bin/accelerate launch --main_process_port 29593 \
        train_depth.py \
        --pretrained_model_name_or_path $MODEL_PATH\
        --output_dir $OUTPUT_DIR \
        --config configs/train/Adapter-XL-depth.yaml \
        --mixed_precision="fp16" \
        --resolution=512 \
        --learning_rate=1e-5 \
        --max_train_steps=100000 \
        --gradient_accumulation_steps=1 \
        --report_to="wandb" \
        --seed=42 \
        --num_train_epochs 100 \
