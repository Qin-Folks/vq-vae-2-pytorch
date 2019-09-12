import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from data_classes.multiple_mnist import MultiMNIST
from utils import utils_io
from vqvae import VQVAE
from scheduler import CycleScheduler
import os

cur_time = utils_io.get_current_time()
exp_dir = os.path.join('experiments', cur_time)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    samples_dir = os.path.join(exp_dir, 'sample')
    models_dir = os.path.join(exp_dir, 'checkpoint')
    os.makedirs(samples_dir)
    os.makedirs(models_dir)


def train(epoch, loader, model, optimizer, scheduler, device):

    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if i % 100 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            # print('max of sample: ', torch.max(sample), torch.min(sample))
            # print('min of out: ', torch.max(out), torch.min(out))
            utils.save_image(
                torch.cat([sample, out], 0),
                f'experiments/{cur_time}/sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
                pad_value=0.5,
            )

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    # parser.add_argument('path', type=str)

    args = parser.parse_args()
    args.size = 64

    print(args)

    device = 'cuda'

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(args.size),
    #         # transforms.CenterCrop(args.size),
    #         transforms.ToTensor(),
    #         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )

    # dataset = datasets.ImageFolder(args.path, transform=transform)
    img_sz = 64
    num_dig = 2
    channels = 1
    to_sort_label = False
    dig_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    nxt_dig_prob = 0.5
    rand_dig_combine = False
    split_dig_set = False
    batch_size = 256
    img_chn = 1
    dataset = MultiMNIST(train=True, data_root='data', image_size=img_sz, num_digits=num_dig,
                         channels=channels, to_sort_label=to_sort_label, dig_to_use=dig_to_use,
                         nxt_dig_prob=nxt_dig_prob, rand_dig_combine=rand_dig_combine,
                         split_dig_set=split_dig_set,
                         )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = nn.DataParallel(VQVAE(in_channel=img_chn)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)
        torch.save(
            model.module.state_dict(), f'experiments/{cur_time}/checkpoint/vqvae_{str(i + 1).zfill(3)}.pt'
        )
