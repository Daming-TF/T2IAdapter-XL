import json
import requests
from PIL import Image
import time
import cv2
import argparse
from logger import logger
import os
import shutil
import numpy as np
import base64
import io
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--target_h", type=int, default=768)
parser.add_argument("--target_w", type=int, default=768)
parser.add_argument("--output_path", help="output image path", required=True)
parser.add_argument("--params_path", type=str, help="params json", default="./temp.json")
parser.add_argument("--output_json", type=str, help="output status json", default="./status.json")
parser.add_argument("--port", type=str, help="port", default="7861")

# my add
parser.add_argument("--input_image", nargs='+', type=str)
parser.add_argument("--input_prompt", type=str)

args = parser.parse_args()


def pil_to_base64(pil_image):
    with io.BytesIO() as stream:
        pil_image.save(stream, "PNG", pnginfo=None)
        base64_str = str(base64.b64encode(stream.getvalue()), "utf-8")
        return "data:image/png;base64," + base64_str


def check_params(params_path, output_path, max_resolution=8000):
    code = 0
    error_str = ""

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.exists(output_dir):
        code = 4002
        error_str = "output dir %s is not exists" % output_dir
        return {'code': code, 'message': error_str}, None, None

    params = {}
    if params_path != "":
        params = json.load(open(args.params_path, "r"))

        if "steps" in params:
            try:
                steps = int(float(params['steps']))
            except Exception as e:
                error_str = "steps is not int"
                code = 4003
                return {'code': code, 'message': error_str}, None, None
        if "cfg_scale" in params:
            try:
                cfg_scale = float(params['cfg_scale'])
            except Exception as e:
                error_str = "cfg_scale is not float"
                code = 4004
                return {'code': code, 'message': error_str}, None, None
        if "height" in params:
            try:
                steps = int(float(params['height']))
            except Exception as e:
                error_str = "height is not int"
                code = 4006
                return {'code': code, 'message': error_str}, None, None
        if "width" in params:
            try:
                steps = int(float(params['width']))
            except Exception as e:
                error_str = "width is not int"
                code = 4007
                return {'code': code, 'message': error_str}, None, None
        if "seed" in params:
            if type(params['seed']) != int:
                error_str = "seed is not int"
                code = 4007
                return {'code': code, 'message': error_str}, None, None
        if "retry" in params:
            if type(params['retry']) != int:
                error_str = "retry is not int"
                code = 4007
                return {'code': code, 'message': error_str}, None, None
        if "prompt" in params:
            if type(params['prompt']) != str:
                error_str = "prompt is not string"
                code = 4008
                return {'code': code, 'message': error_str}, None, None
        if "negative_prompt" in params:
            if type(params['negative_prompt']) != str:
                error_str = "negative_prompt is not string"
                code = 4009
                return {'code': code, 'message': error_str}, None, None
        if "stages" in params:
            if len(params['stages']) > 0:
                denoising_strengths = params['stages']['denoising_strengths']
                step_ratios = params['stages']['step_ratios']
                if len(denoising_strengths) != len(step_ratios):
                    error_str = "the length of denoising_strengths is not equal to step_ratios"
                    code = 4010
                    return {'code': code, 'message': error_str}, None, None

    return {'code': code, 'message': error_str}, params


# @logger.catch(reraise=True)
def test_server(args, status_dict, params):
    if status_dict['code'] != 0:  # check 2, 7
        logger.error('Check params failed: code = {}'.format(status_dict['code']))
        try:
            shutil.copyfile(args.input_path, args.output_path)
        except Exception as e:
            status_dict['message'] += " and copy input to output fail"
            logger.error('Copy input to output failed!')
        with open(args.output_json, 'w') as f:
            json.dump(status_dict, f)
        return

    logger.info('Read port')
    url = "http://127.0.0.1:%s" % args.port
    # start_time = time.time()
    # print("start infer")
    # limit size to maxium resolution
    input_h = args.target_h
    input_w = args.target_w

    ratio = input_h / input_w
    if ratio > 1:
        input_h = 768
        input_w = int(768 / ratio)
    else:
        input_w = 768
        input_h = int(768 * ratio)

    # check if has control net
    if "alwayson_scripts" in params:
        for control in params["alwayson_scripts"]["controlnet"]["args"]:
            control["processor_res"] = min(input_h, input_w)

    stages = {}
    if "specifics" in params:
        specifics = params['specifics']
    if "stages" in params:
        stages = params['stages']

    params["output_path"] = args.output_path

    payload = {
        "seed": -1,
        "height": input_h,
        "width": input_w,
        "retry": 1,
        "stages": stages,
    }
    payload.update(params)

    if "prompt_distribution" in params:
        prompt_distribution = params["prompt_distribution"]
        prob = []
        attris = []
        for key in prompt_distribution:
            attris.append(key)
            prob.append(prompt_distribution[key])
        normal_prob = [p / sum(prob) for p in prob]
        value = np.random.choice(attris, p=normal_prob)
        payload["prompt"] += ", " + value

    if "negprompt_distribution" in params:
        negprompt_distribution = params["negprompt_distribution"]
        prob = []
        attris = []
        for key in negprompt_distribution:
            attris.append(key)
            prob.append(negprompt_distribution[key])
        normal_prob = [p / sum(prob) for p in prob]
        value = np.random.choice(attris, p=normal_prob)
        payload["negative_prompt"] += ", " + value

    payload['steps'] = int(float(payload['steps']))
    payload['cfg_scale'] = float(payload['cfg_scale'])
    payload['height'] = int(float(payload['height']))
    payload['width'] = int(float(payload['width']))
    payload_json = json.dumps(payload)
    try:
        logger.info('Request txt2img')
        response = requests.post(url=f'{url}/sdapi/v1/txt2img', data=payload_json).json()

        result = response['images'][0]
        image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
        if input_w <= args.target_w:
            image = image.resize((args.target_w, args.target_h), Image.BILINEAR)
        else:
            image = image.resize((args.target_w, args.target_h), Image.LANCZOS)
        # image.save(args.output_path)
        return image

    except Exception as e:
        code = 5001
        logger.error(f'Cannot request to txt2img!')
        with open(args.output_json, 'w') as f:
            json.dump({'code': code, 'message': repr(e)}, f)
        return

        # logger.info(response)
    logger.info('Time elapsed: {}'.format(time.time() - start_time))

    with open(args.output_json, 'w') as f:
        json.dump({'code': 0, 'message': "SUCCESS"}, f)

    logger.info('======== End Server ========')
    return


if __name__ == "__main__":
    # init
    logger.info('======== Start Server ========')
    start_time = time.time()
    logger.info('Check params')
    status_dict, params = check_params(args.params_path, args.output_path)
    controlnet_params = params['alwayson_scripts']['controlnet']['args'][0]
    save_dir = args.output_path
    os.makedirs(save_dir, exist_ok=True)
    print(f'save_dir:{save_dir}')

    # prepare data
    # 1.
    scales = list(np.arange(0, 1.1, 0.1))
    # 2.
    # ToDo:两组不同风格的lora(人像)
    image_dir = r'/home/mingjiahui/data/ipadapter/test_data/lora_test'
    image_paths = [os.path.join(image_dir, name)for name in os.listdir(image_dir)]
    # 3.
    prompts = [
        # '1old lady',
        '1lady'
    ]
    # 4.
    lora_list = [
        # '<lyco:Paper_Cutout:1.0>',
        # '<lyco:Chinese_Aesthetic_Illustration:1.0>',
        # '<lyco:Paper_Cutout:1.0><lyco:Chinese_Aesthetic_Illustration:1.0>',
        # '',
        '<lyco:Paper_Cutout:1.0><lyco:8Bit_Scenes:0.6><lyco:8Bit_Objects:0.3>'
    ]
    fix_prompts = [
        # ',seekoo_gds,paper illustration,flat illustration,16k,masterpiece,best quality,sharp,',
        ',seekoo_gds, oriental style painting, Chinese painting, flowing forms,',
        # ',seekoo_gds,paper illustration,flat illustration,16k,masterpiece,best quality,sharp',
        # 'paper illustration,flat illustration,16k,masterpiece,best quality,sharp',
    ]
    negative_prompts = [
        # 'BadDream, noise',
        'BadDream',
        # 'BadDream, noise',
        # 'BadDream',
    ]
    assert len(lora_list)==len(fix_prompts)==len(negative_prompts)

    # process
    for lora_key, fix_prompt, negative_prompt in zip(lora_list, fix_prompts, negative_prompts):
        # get save dir
        lora_id = lora_key.replace('lyco:', '').replace('><', '-').replace('<', '').replace('>', '').replace(':', '_') \
            if lora_key != '' else 'no_lora'
        save_dir_ = os.path.join(save_dir, lora_id)
        os.makedirs(save_dir_, exist_ok=True)

        params['negative_prompt'] = negative_prompt
        for prompt in prompts:
            params['prompt'] = prompt + fix_prompt + lora_key
            for image_path in image_paths:
                image_id = os.path.basename(image_path).split('.')[0]
                controlnet_params['input_image'] = image_path
                h_concat = None
                for scale in scales:
                    controlnet_params['weight'] = scale
                    # connect to server
                    output = np.array(test_server(args, status_dict, params))
                    h_concat = cv2.hconcat([h_concat, output]) if h_concat is not None else output

                Image.fromarray(h_concat).save(os.path.join(save_dir_, f'{image_id}-{prompt}.jpg'))

    # out_put = test_server(args, status_dict, params)
    # out_put.save(r'./output/debug.jpg')