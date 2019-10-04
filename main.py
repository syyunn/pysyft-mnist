import time

import torch
import torch.optim as optim

import syft as sy

import args
import connections as conn
import loader

from train import train
from model import Model


def main():
    torch.manual_seed(args.seed)

    hook = sy.TorchHook(torch)

    workers = conn.connect_to_workers(hook=hook, n_workers=2)
    crypto_provider = conn.connect_to_crypto_provider(hook=hook)

    private_train_loader, private_test_loader = loader.get_private_data_loaders(
        workers=workers,
        crypto_provider=crypto_provider)
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train(args=args,
          model=model,
          private_train_loader=private_train_loader,
          optimizer=optimizer,
          epoch=args.epochs)


if __name__ == "__main__":
    main()
