"""
Ref = https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/utils.py
"""

import torch
from torchvision import transforms

import utils
from transformer_net import TransformerNet

CONTENT_IMG_PATH = "trainmini/000000000072.jpg"
MODEL = "checkpoints/ckpt_epoch_0_batch_id_1150.pth"
OUTPUT_IMG_PATH = "outputs/trainimg18.jpg"


def stylize():
    device = torch.device("mps")

    content_image = utils.load_image(CONTENT_IMG_PATH)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(MODEL, map_location=device)
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
    utils.save_image(OUTPUT_IMG_PATH, output[0])
    utils.save_image("outputs/van_img.jpg", content_image[0])


stylize()
