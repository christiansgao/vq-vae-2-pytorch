import argparse
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.vqvae import VQVAE
from src.pixelsnail import PixelSNAIL
from src.Paths import Paths

########################################################
# Creates sample of decoded model and saves as png image
########################################################

@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def load_model(model, checkpoint, device, channel):
    checkooint_path = os.path.join('checkpoint', checkpoint)
    ckpt = torch.load(checkooint_path)

    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [32, 32],
            512,
            channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )
        
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--vqvae', type=str, default=Paths.CHECKPOINT_PRETRAINED)
    parser.add_argument('--top', type=str, default=Paths.CHECKPOINT_PRETRAINED)
    parser.add_argument('--bottom', type=str, default=Paths.CHECKPOINT_PRETRAINED)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--filename', type=str, default= "test.png")

    args = parser.parse_args()

    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

    top_sample = sample_model(model_top, device, args.batch, [32, 32], args.temp)
    bottom_sample = sample_model(
        model_bottom, device, args.batch, [64, 64], args.temp, condition=top_sample
    )

    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    decoded_sample = decoded_sample.clamp(-1, 1)

    save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))
