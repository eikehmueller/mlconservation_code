"""Check that the neural network Lagrangian is invariant und rotations
and coordinate shifts
"""

import pytest
import numpy as np
from nn_models import XYModelNNLagrangian, LagrangianModel


@pytest.mark.parametrize("dim", [2, 4, 6, 8])
def test_xymodel_nn_lagrangian_rotation_invariance(dim):
    """Check that the neural network Lagrangian has the same value
    if all angles are shifted by a fixed amount.

    In other words, this verifies that

      L(theta_0+phi,...,theta_{d-1}+phi,dot(theta)_0,...,dot(theta)_{d-1})
      = L(theta_0,...,theta_{d-1},dot(theta)_0,...,dot(theta)_{d-1})

    :arg dim: dimension of system
    """
    nn_lagrangian = XYModelNNLagrangian(
        dim, rotation_invariant=True, shift_invariant=True
    )
    # number of samples to check
    n_samples = 4
    # rotation angle
    phi = 1.1
    # tolerance for tests
    tolerance = 1.0e-6
    q = np.random.uniform(low=-np.pi, high=+np.pi, size=(n_samples, dim))
    qdot = np.random.normal(size=(n_samples, dim))
    X = np.concatenate([q, qdot], axis=1)
    X_rotated = np.concatenate([q + phi, qdot], axis=1)
    dL = nn_lagrangian(X_rotated) - nn_lagrangian(X)
    assert np.linalg.norm(dL) < tolerance


@pytest.mark.parametrize("dim", [4, 6, 8])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_xymodel_nn_lagrangian_shift_invariance(dim, offset):
    """Check that the neural network Lagrangian has the same value
    if all angles are shifted by a fixed offset.

    In other words, this verifies that

      L(theta_0,theta_1,...,theta_{d-1},dot(theta)_0,dot(theta)_1,...,dot(theta)_{d-1})
      = L(theta_1,...,theta_{d-1},theta_0,dot(theta)_1,...,dot(theta)_{d-1},dot(theta)_0)

    :arg dim: dimension of system
    :arg offset: shift offset
    """
    nn_lagrangian = XYModelNNLagrangian(
        dim, rotation_invariant=True, shift_invariant=True
    )
    # number of samples to check
    n_samples = 4
    # rotation angle
    phi = 1.1
    # tolerance for tests
    tolerance = 1.0e-6
    q = np.random.uniform(low=-np.pi, high=+np.pi, size=(n_samples, dim))
    qdot = np.random.normal(size=(n_samples, dim))
    X = np.concatenate([q, qdot], axis=1)
    X_shifted = np.concatenate(
        [np.roll(q, offset, axis=1), np.roll(qdot, offset, axis=1)], axis=1
    )
    dL = nn_lagrangian(X_shifted) - nn_lagrangian(X)
    assert np.linalg.norm(dL) < tolerance
