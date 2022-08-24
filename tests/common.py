"""Common functionality used by all tests"""
import os
import numpy as np
import pytest

import tensorflow as tf

# Use double precision for tests
tf.keras.backend.set_floatx("float64")

# Suppress tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
    return np.asarray(M_mat, dtype=np.float64), np.asarray(A_mat, dtype=np.float64)


@pytest.fixture
def rng():
    """Return random generator fixture that can be used by all tests"""
    return np.random.default_rng(21494917)
