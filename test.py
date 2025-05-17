import requests
import torch
import os

from PIL import Image
from pathlib import Path
from prismatic import load
from prismatic.models.backbones.vision.vggt import VGGTBackbone
from vggt.models.vggt import VGGT

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = ""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)
model_id = "dinov2-224px+7b"
vlm = load(model_id, hf_token=hf_token)

# Add VGGT backbone directly
vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
vggt_backbone = VGGTBackbone(vggt_model)
vlm.vggt_backbone = vggt_backbone
vlm.vggt_backbone.to(device)

# Move model to device
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# save image
image.save("image.png")
image_path = "image.png"

user_prompt = "What is going on in this image?"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()
print(prompt_text)

# Generate!
generated_text = vlm.generate(
    image=image,
    prompt_text=prompt_text,
    image_paths=image_path,  # Add image path for VGGT
    do_sample=True,
    temperature=0.4,
    max_new_tokens=512,
    min_length=8,
)
print(generated_text)
