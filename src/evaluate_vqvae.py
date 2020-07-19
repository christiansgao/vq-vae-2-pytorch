import argparse
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

import distributed as dist
from src.Paths import Paths
from src.vqvae import VQVAE

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(checkooint):
    ckpt = torch.load(checkooint, map_location=torch.device(DEVICE))
    model = VQVAE()
    model.load_state_dict(ckpt)
    model = model.to(DEVICE)
    model.eval()

    return model


def evaluate(loader, model, out_path):
    if dist.is_primary():
        loader = tqdm(loader)

    sample_size = 25
    model.eval()

    i, (img, label) = next(enumerate(loader))

    sample = img[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)

    utils.save_image(
        torch.cat([sample, out], 0),
        out_path,
        nrow=sample_size,
        normalize=True,
        range=(-1, 1),
    )

    # recon_loss = criterion(out, img)
    # print("Loss: ",recon_loss)


def main(args):
    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = load_model(args.checkpoint).to(DEVICE)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )
    evaluate(loader, model, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--path", type=str, default=Paths.TRAINING)
    parser.add_argument("--checkpoint", type=str, default=Paths.CHECKPOINT)
    parser.add_argument("--out_path", type=str, default=Paths.EVAL_OUTPUT)

    parser.add_argument("--sched", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
