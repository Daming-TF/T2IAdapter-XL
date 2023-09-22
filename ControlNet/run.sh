export CUDA_VISIBLE_DEVICES=$1
export INPUT1='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2'
export INPUT2='/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2/14_v2.jpg'
export OUTPUT='/home/mingjiahui/data/ControlNet/'
export COND='/home/mingjiahui/data/T2IAdapter/offical/sketch/'
python ControlNet/inference_test_lineart_reconstruction.py \
      --input=$INPUT1 \
      --output=$2 \
      --batch_size=4 \
      --seed=42 \
      --resolution=1024 \
      --step=50 \
      --scale=$3 \
      --color_inversion=False \
      --cond=$COND
#      --prompt='Two guinea pigs，side by side, Christmas hat, in front of a Christmas tree'

