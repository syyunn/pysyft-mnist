import torch
import torch.utils.data
from torchvision import datasets, transforms

import utils
import args


def get_private_data_loaders(workers, crypto_provider):
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transformation),
        batch_size=args.batch_size
    )

    private_train_loader = []
    for idx, (mnist, target) in enumerate(train_loader):
        if idx < args.n_train_items / args.batch_size:
            secret_mnist = utils.share_tensor_in_secret(tensor=mnist,
                                                        precision_fractional=
                                                        args.
                                                        precision_fractional,
                                                        workers=workers,
                                                        crypto_provider=
                                                        crypto_provider)
            secret_target = utils.share_tensor_in_secret(tensor=target,
                                                         precision_fractional=
                                                         args.
                                                         precision_fractional,
                                                         workers=workers,
                                                         crypto_provider=
                                                         crypto_provider)
            private_train_loader.append((secret_mnist, secret_target))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=transformation),
        batch_size=args.test_batch_size
    )

    private_test_loader = []
    for idx, (mnist, target) in enumerate(test_loader):
        if idx < args.n_test_items / args.batch_size:
            secret_mnist = utils.share_tensor_in_secret(tensor=mnist,
                                                        precision_fractional=
                                                        args.
                                                        precision_fractional,
                                                        workers=workers,
                                                        crypto_provider=
                                                        crypto_provider)
            secret_target = utils.share_tensor_in_secret(tensor=target,
                                                         precision_fractional=
                                                         args.
                                                         precision_fractional,
                                                         workers=workers,
                                                         crypto_provider=
                                                         crypto_provider)
            private_test_loader.append((secret_mnist, secret_target))

    return private_train_loader, private_test_loader
