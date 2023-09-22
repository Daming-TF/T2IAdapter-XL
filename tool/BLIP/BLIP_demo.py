import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

vision = '/mnt/nfs/file_server/public/mingjiahui/models/Salesforce--blip-image-captioning-base'     # "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(vision)
model = BlipForConditionalGeneration.from_pretrained(vision).to("cuda")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
raw_image.save(r'/home/mingjiahui/data/debug.jpg')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# >>> a woman sitting on the beach with her dog
