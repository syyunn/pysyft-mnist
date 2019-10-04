import torch


def make_one_hot_label(index_tensor):
    """
    Transform to one hot tensor

    Example:
        [0, 3, 9]
        =>
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]

    """
    one_hot_tensor = torch.zeros(*index_tensor.shape, 10)
    one_hot_tensor = one_hot_tensor.scatter(1, index_tensor.view(-1, 1), 1)

    return one_hot_tensor


def share_tensor_in_secret(tensor, precision_fractional, workers,
                           crypto_provider):
    """
    Transform to fixed precision and secret share a tensor
    """
    shared = tensor.fix_precision(
        precision_fractional=precision_fractional).share(
        *workers, crypto_provider=crypto_provider, requires_grad=True)

    return (shared)
