#export CUDA_VISIBLE_DEVICES=7,
# sd1_5-control_test.py
python3 ./tool/webui/sd1_5-control_test_v2.py \
        --output_path=/home/mingjiahui/data/result/test3 \
        --params_path=./tool/webui/json/sketch_demo.json \
        --port 7879 \
#        --params_path=./tool/webui/json/ipadapter_test.json  \
#        --input_prompt=./tool/webui/test_data/prompt.txt \
#        --input_image ./tool/webui/test_data/chinese_aesthetic.txt \
#                      ./tool/webui/test_data/colorful_rhythm.txt \
#                      ./tool/webui/test_data/paper_cutout_images.txt \

