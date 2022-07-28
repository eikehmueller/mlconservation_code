"""Check that the neural network Lagrangian is invariant und rotations
and coordinate shifts
"""

import pytest
import numpy as np
from time_integrator import RK4Integrator
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


@pytest.mark.skip(reason="too expensive")
def test_xymodel_nn_eigenstate_shift_invariance():
    """Check that that if the dynamical system is initialised with an eigenstate
    of the shift operator, the dynamics will will preserve shift invariance.

    For this, set for j = 0,1,...,7

      theta_j(0)      = alpha_1*cos(pi/4*j + delta_1)
      dot(theta)_j(0) = alpha_2*cos(pi/4*j + delta_2)

    which satisfies

      theta_{j+4}(0) = -thetaj(0) and S^4 dot(theta)_{j+4}(0) = -dot(theta)_j(0).

    Since the Lagrangian is shift-invariant, this will also hold at
    later times:

      theta_{j+4}(t) + theta_j(t) = 0
    """
    # dimension of dynamical system (number of spins)
    dim = 4
    # timestep size
    dt = 0.01
    # final time
    t_final = 1.0
    nn_lagrangian = XYModelNNLagrangian(
        dim, rotation_invariant=True, shift_invariant=True
    )
    model = LagrangianModel(nn_lagrangian)
    time_integrator = RK4Integrator(model, dt)
    alpha_1 = 1.3
    alpha_2 = 2.4
    delta_1 = 0.3
    delta_2 = 1.7
    # tolerance for tests
    tolerance = 1.0e-6
    q0 = np.asarray(
        alpha_1 * np.cos(0.5 * np.pi * np.arange(dim) + delta_1), dtype=np.float32
    )
    qdot0 = np.asarray(
        alpha_2 * np.cos(0.5 * np.pi * np.arange(dim) + delta_2), dtype=np.float32
    )
    time_integrator.set_state(q0, qdot0)
    nsteps = int(t_final / dt)
    time_integrator.integrate(nsteps)
    q = time_integrator.q
    print("q = ", q)
    assert np.linalg.norm(q + np.roll(q, shift=dim // 4, axis=0)) < tolerance
