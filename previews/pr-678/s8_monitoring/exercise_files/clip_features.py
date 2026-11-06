import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True, timeout=30).raw)

# set either text=None or images=None when only the other is needed
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

img_features = model.get_image_features(inputs["pixel_values"])
text_features = model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
print(img_features.shape, text_features.shape)
