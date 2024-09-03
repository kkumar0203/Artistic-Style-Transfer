"""
Ref = https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/utils.py
"""

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import time
import os
import utils
from transformer_net import TransformerNet
from vgg import Vgg16

IMG_SIZE = 256
BATCH_SIZE = 4
LR = 1e-3
DATASET_PATH = "../style_transfer/code/images/train"
STYLE_IMG_PATH = "./style-images/tsunami.png"
EPOCHS = 2
CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 1e10
LOG_INTERVAL = 10
CHECKPOINT_MODEL_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 1000
SAVE_MODEL_DIR = "./models"


def train():
    device = torch.device("mps")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), lr=LR)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(STYLE_IMG_PATH)
    style = style_transform(style)
    style = style.repeat(BATCH_SIZE, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(EPOCHS):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = CONTENT_WEIGHT * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= STYLE_WEIGHT

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % LOG_INTERVAL == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if CHECKPOINT_MODEL_DIR is not None and (batch_id + 1) % CHECKPOINT_INTERVAL == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(CHECKPOINT_MODEL_DIR, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(EPOCHS) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        CONTENT_WEIGHT) + "_" + str(STYLE_WEIGHT) + ".model"
    save_model_path = os.path.join(SAVE_MODEL_DIR, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


train()
