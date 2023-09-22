from PIL import Image
import numpy as np
import requests
import torch

from transformers import DPTForDepthEstimation, DPTFeatureExtractor

model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test/000000000285.jpg')
image.save(r'/home/mingjiahui/data/depth_test.png')

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

print(prediction.shape)
# visualize the prediction
output = prediction.squeeze().cpu().numpy()
print(output.shape)
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
depth.save(r'/home/mingjiahui/data/depth.png')
# depth.show()
