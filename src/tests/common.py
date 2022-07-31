import numpy as np
import pytest

"""Common functionality used by all tests"""


def harmonic_oscillator_matrices(dim):
    """Construct matrices M and A for
    harmonic oscillator

    :arg dim: dimension of state space
    """
    np.random.seed(2151827)
    M_tmp = np.random.normal(size=(dim, dim))
    M_mat = M_tmp.T @ M_tmp + 0.1 * np.identity(dim)
    A_tmp = np.random.normal(size=(dim, dim))
    A_mat = A_tmp.T @ A_tmp + 0.1 * np.identity(dim)
    return np.array(M_mat, dtype=np.float32), np.array(A_mat, dtype=np.float32)


@pytest.fixture
def random_seed():
    """Return common random seed to be used by all tests"""
    return 21494917
