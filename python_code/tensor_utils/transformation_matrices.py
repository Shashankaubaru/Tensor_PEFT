import torch
import numpy as np
import scipy.fftpack
import scipy


def random_orthogonal(dim3, dtype=None, device=None):
    q = torch.linalg.qr(torch.randn(dim3, dim3), mode='complete')[0]
    q = q.to(dtype=dtype, device=device).squeeze().contiguous()
    return q


def dct_matrix(dim, dtype=None, device=None):
    """
    Form orthogonal dct matrix for transformations
    :param dim: size of transformation matrix
    :return: dim x dim orthogonal transformation matrix
    """
    C = scipy.fftpack.dct(np.eye(dim), norm="ortho")
    C = np.transpose(C)

    return torch.tensor(C, dtype=dtype, device=device).contiguous()


def dft_matrix(dim, dtype=None, device=None):
    """
    Form orthogonal DFT matrix for transformations
    :param dim: size of transformation matrix
    :return: dim x dim orthogonal transformation matrix
    """

    C = scipy.fft.fft(np.eye(dim))
    C = np.transpose(C)

    return torch.tensor(C, dtype=dtype, device=device).contiguous()