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

parser = argparse.ArgumentParser()
parser.add_argument("--target_h", type=int, default=768)
parser.add_argument("--target_w", type=int, default=768)
parser.add_argument("--output_path", help="output image path", required=True)
parser.add_argument("--params_path", type=str, help="params json", default="./temp.json")
parser.add_argument("--output_json", type=str, help="output status json", default="./status.json")
parser.add_argument("--port", type=str, help="port", default="7861")

# # my add
# parser.add_argument("--input_image", nargs='+', type=str)
# parser.add_argument("--input_prompt", type=str)

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
        image.save(args.output_path)
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


import re
def replace_lora_key(old_prompt, extra_prompt, lora_key):
    pattern = r'<[^>]*>'
    old_prompt = re.sub(pattern, '', old_prompt)
    new_prompt = extra_prompt + old_prompt + lora_key
    return new_prompt


if __name__ == "__main__":
    logger.info('======== Start Server ========')
    start_time = time.time()
    logger.info('Check params')
    status_dict, params = check_params(args.params_path, args.output_path)
    controlnet_params = params['alwayson_scripts']['controlnet']['args'][0]

    prompt_list = [
        '1fish',
        '3fish',
        'sun,the roof, the hallway',
        'computer, desk, mouse, keyboard',
        '1man, the shadow, In the light',
        'high buildings, large mansions',
        'mansion in the woods',
        'sunrise, sailboat, seaside',
        'sun, garden',
        'telephone poles, cars, railroad,',
        '1dog'
    ]

    image_paths = [
        r'/home/mingjiahui/data/test_data/sketch_10/img_v2_d8f32e2d-c587-444b-9147-68a276db6b0g.jpg',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321975.png',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321976.png',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321977.png',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321978.png',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321979.png',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321980.png',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321981.png',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321982.png',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321983.png',
        r'/home/mingjiahui/data/test_data/sketch_10/Frame 427321984.png',
    ]
    assert len(prompt_list)==len(image_paths)

    controlnet_select = [
        {'t2ia_sketch_pidi': 't2iadapter_sketch_sd15v2'},
        # {'none': 't2iadapter_sketch_sd15v2'},
        {'scribble_pidinet': 'control_v11p_sd15_scribble'},
        # {'none': 'control_v11p_sd15_scribble'},
    ]

    lora_list = [
        '<lyco:Paper_Cutout:1.0>',
        '<lyco:8Bit_Scenes:0.6><lyco:8Bit_Objects:0.3>',
        # '',
    ]
    fix_prompts = [
        ',seekoo_gds,paper illustration,flat illustration,16k,masterpiece,best quality,sharp,',
        ',seekoo_gds,paper illustration,flat illustration,16k,masterpiece,best quality,sharp',
        # 'paper illustration,flat illustration,16k,masterpiece,best quality,sharp'
    ]

    negative_prompts = [
        'BadDream,low quality,low resolution,bad art,poor detailing,ugly,disfigured,text,watermark,signature,bad proportions,bad anatomy,duplicate,cropped,cut off,extra hands,extra arms,extra legs,poorly drawn face,unnatural pose,out of frame,unattractive,twisted body,extra limb,missing limb,mangled,malformed limbs,',
        'BadDream, noise',
        # 'BadDream, noise',
    ]

    control_mode = {
        0: 'balanced',
        # 1: 'prompt_is_important',
        # 2: 'controlnet_is_import',
    }
    weights = np.arange(0.6, 1.1, 0.1).tolist()

    save_dir = args.output_path
    os.makedirs(save_dir, exist_ok=True)
    for lora_key, fix_prompt, negative_prompt in zip(lora_list, fix_prompts, negative_prompts):
        img_save_dir_0 = os.path.join(save_dir, lora_key) if lora_key != '' else os.path.join(save_dir, 'no_lora')
        os.makedirs(img_save_dir_0, exist_ok=True)
        params['negative_prompt'] = negative_prompt

        for module_model_pair in controlnet_select:
            module, model = next(iter(module_model_pair.items()))
            img_save_dir_1 = os.path.join(img_save_dir_0, f'{model}-{module}')
            os.makedirs(img_save_dir_1, exist_ok=True)
            controlnet_params['module'] = module
            controlnet_params['model'] = model

            for mode_id, mode_v in control_mode.items():
                img_save_dir_2 = os.path.join(img_save_dir_1, f'{mode_id}-{mode_v}')
                os.makedirs(img_save_dir_2, exist_ok=True)
                controlnet_params['control_mode']=mode_id

                for weight in weights:
                    img_save_dir_3 = os.path.join(img_save_dir_2, '{:.2f}'.format(weight))
                    os.makedirs(img_save_dir_3, exist_ok=True)
                    controlnet_params['weight'] = weight

                    for image_path, prompt in zip(image_paths, prompt_list):
                        seed = 41
                        for _ in range(1):
                            seed = seed + 1
                            params['prompt'] = prompt + fix_prompt + lora_key
                            params['seed'] = seed
                            controlnet_params['input_image'] = image_path

                            # save name
                            image_id = os.path.basename(image_path).split('.')[0]
                            save_name = f"{image_id}-{seed}.jpg"
                            args.output_path = os.path.join(img_save_dir_3, save_name)

                            print(f'***********************\n'
                                  f"* image:\t\t{controlnet_params['input_image']}\n"
                                  f"* prompt:\t\t{params['prompt']}\n"
                                  f"* seed:\t\t{params['seed']}\n"
                                  f"* save_path:\t\t{args.output_path}\n"
                                  f"* control parmar:\n"
                                  f" ** module:\t\t{controlnet_params['module']}\n"
                                  f" ** model:\t\t{controlnet_params['model']}\n"
                                  f" ** control model:\t\t{controlnet_params['control_mode']}-{mode_v}\n"
                                  f" ** weight:\t\t{controlnet_params['weight']}\n"
                                  )


                            test_server(args, status_dict, params)

            # merge_img()
