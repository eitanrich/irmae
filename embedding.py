#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import mixture
import matplotlib.pyplot as plt
import model
import utils
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Generative and Downstream Tasks")
parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('-n', type=int, help='latent dimension', default=128)
parser.add_argument('-l', type=int, help='layers', default=0)
parser.add_argument('-X', type=int, default=10)
parser.add_argument('-Y', type=int, default=10)
parser.add_argument('-N', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--model-name', type=str, default="default")
parser.add_argument('--vae', action='store_true', help='VAE')
parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
parser.add_argument('--data-path', type=str, default="./data/")
parser.add_argument('--save-path', type=str, default="./results/")


def main(args):

    # seed ##############
    np.random.seed(2020)
    torch.manual_seed(1)

    # dataset ##########################################
    if args.dataset == "mnist":
        args.data_path = args.data_path + "mnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        test_set = datasets.MNIST(args.data_path, train=False,
                                  download=True,
                                  transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(
            test_set,
            shuffle=False,
            num_workers=32,
            batch_size=args.batch_size
        )
    elif args.dataset == "shape":
        test_set = utils.ShapeDataset(data_size=args.batch_size)
        test_set.set_seed(2020)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            shuffle=False,
            num_workers=32,
            batch_size=args.batch_size
        )
    elif args.dataset == "celeba":
        test_set = utils.ImageFolder(
            args.data_path+'/', # + '/test/',
            transform=transforms.Compose([  # transforms.CenterCrop(148),
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        print(args.data_path, len(test_set))
        test_loader = torch.utils.data.DataLoader(
            test_set,
            shuffle=False,
            num_workers=32,
            batch_size=args.batch_size
        )

    # load model ##########################################

    if args.dataset == "mnist":
        if args.vae:
            enc = model.MNIST_Encoder(args.n * 2)
            dec = model.MNIST_Decoder(args.n, vae=True)
        else:
            enc = model.MNIST_Encoder(args.n)
            dec = model.MNIST_Decoder(args.n)
    elif args.dataset == "celeba":
        if args.vae:
            enc = model.CelebA_Encoder(args.n * 2)
            dec = model.CelebA_Decoder(args.n, vae=True)
        else:
            enc = model.CelebA_Encoder(args.n)
            dec = model.CelebA_Decoder(args.n)
    elif args.dataset == "shape":
        if args.vae:
            enc = model.Shape_Encoder(args.n * 2)
            dec = model.Shape_Decoder(args.n, vae=True)
        else:
            enc = model.Shape_Encoder(args.n)
            dec = model.Shape_Decoder(args.n)

    dec.load_state_dict(torch.load(
        args.checkpoint + "/" + args.dataset + "/dec_" + args.model_name,
        map_location=torch.device('cpu')))
    enc.load_state_dict(torch.load(
        args.checkpoint + "/" + args.dataset + "/enc_" + args.model_name,
        map_location=torch.device('cpu')))
    dec.eval()
    enc.eval()

    if args.l > 0:
        mlp = model.MLP(args.n, args.l)
        mlp.load_state_dict(torch.load(
                args.checkpoint + "/" + args.dataset +
                "/mlp_" + args.model_name,
                map_location=torch.device('cpu')))
        mlp.eval()

    #####################################################

    all_z = []
    for yi, _ in test_loader:
        if args.vae:
            z_hat = enc(yi)
            mu = z_hat[:, :args.n]
            logvar = z_hat[:, args.n:]
            zi = model.reparametrization(mu, logvar)
        else:
            if args.l > 0:
                zi = mlp(enc(yi))
            else:
                zi = enc(yi)
        all_z.append(zi.detach().cpu().numpy())

    path = args.save_path + "/" + args.dataset + "/" + args.model_name
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + "/latents.npy", np.vstack(all_z))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
