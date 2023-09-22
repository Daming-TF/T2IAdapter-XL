export CUDA_VISIBLE_DEVICES=$1
export TestData='/mnt/nfs/file_server/public/mingjiahui/data'
export MyPath='/mnt/nfs/file_server/public/mingjiahui/models'

export BaseModelPath="$MyPath/runwayml--stable-diffusion-v1-5"
#export BaseModelPath="$MyPath/models/webui/dreamshaper_631BakedVae"
export VAEModelPath="$MyPath/stabilityai--sd-vae-ft-mse/"
#export VAEModelPath="$MyPath/models/webui/vae-ft-mse-840000-ema-pruned"

export ImaegeEncoderPath="$MyPath/h94--IP-Adapter/h94--IP-Adapter/models/image_encoder/"
# ip-adapter_sd15.bin
# ip-adapter-plus-face_sd15.bin
# ip-adapter-plus_sd15.bin
export IPCkpt="$MyPath/h94--IP-Adapter/h94--IP-Adapter/models/ip-adapter-plus_sd15.bin"
export ControlnetModelPath="$MyPath/lllyasviel--sd-controlnet-canny"
# ipadapter_demo.py, ip_adapter_controlnet_demo.py
python IPAdapter/experiment/exp_single_img_inference/add_feature_in_diff_denoising.py \
      --base_model_path=$BaseModelPath \
      --vae_model_path=$VAEModelPath \
      --image_encoder_path=$ImaegeEncoderPath \
      --ip_ckpt=$IPCkpt \
      --controlnet_model_path=$ControlnetModelPath \
      --cond_img="$TestData/inference_test_v2/15_v2.jpg" \
      --img_prompt="$TestData/ip-adapter/statue.png"