import os

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

vision = '/mnt/nfs/file_server/public/mingjiahui/models/Salesforce--blip-image-captioning-base'  # "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(vision)
model = BlipForConditionalGeneration.from_pretrained(vision).to("cuda")


def process(image_path):
    raw_image = Image.open(image_path).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    prompt = processor.decode(out[0], skip_special_tokens=True)

    print('test')
    print(f'{image_path}\n{prompt}')
    # txt_path = image_path.replace('.jpg', '.txt')
    # with open(txt_path, 'w') as f:
    #     f.write(prompt)
    #     print(f"Prompt:'{prompt}' success write in ==> {txt_path}")


if __name__ == '__main__':
    image_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2'
    for name in os.listdir(image_dir):
        # print(name)
        if not name.endswith('.jpg'):
            continue
        image_path = os.path.join(image_dir, name)
        if not os.path.exists(image_path.replace('.jpg', '.txt')):
            continue
        process(image_path)



