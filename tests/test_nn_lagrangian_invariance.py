"""Check that the neural network Lagrangian is invariant und rotations
and coordinate shifts
"""

import pytest
import numpy as np
import tensorflow as tf
from scipy.stats import ortho_group, special_ortho_group

from conservative_nn.time_integrator import RK4Integrator
from conservative_nn.nn_lagrangian import (
    XYModelNNLagrangian,
    SingleParticleNNLagrangian,
    TwoParticleNNLagrangian,
    SchwarzschildNNLagrangian,
)
from conservative_nn.nn_lagrangian_model import NNLagrangianModel
from common import rng


@pytest.fixture
def dense_layers():
    """Return dense layers for neural network lagrangians"""
    return [
        tf.keras.layers.Dense(32, activation="softplus"),
        tf.keras.layers.Dense(32, activation="softplus"),
    ]


@pytest.mark.parametrize("dim", [2, 4, 6, 8])
def test_xymodel_nn_lagrangian_rotation_invariance(rng, dense_layers, dim):
    """Check that the neural network Lagrangian has the same value
    if all angles are shifted by a fixed amount.

    In other words, this verifies that

      L(theta_0+phi,...,theta_{d-1}+phi,dot(theta)_0,...,dot(theta)_{d-1})
      = L(theta_0,...,theta_{d-1},dot(theta)_0,...,dot(theta)_{d-1})

    :arg dim: dimension of system
    """
    nn_lagrangian = XYModelNNLagrangian(
        dim, dense_layers, rotation_invariant=True, shift_invariant=True
    )
    # number of samples to check
    n_samples = 4
    # rotation angle
    phi = 1.1
    # tolerance for tests
    tolerance = 2.0e-5
    q = rng.uniform(low=-np.pi, high=+np.pi, size=(n_samples, dim))
    qdot = rng.normal(size=(n_samples, dim))
    X = np.concatenate([q, qdot], axis=1)
    X_rotated = np.concatenate([q + phi, qdot], axis=1)
    dL = nn_lagrangian(X_rotated) - nn_lagrangian(X)
    assert np.linalg.norm(dL) < tolerance


@pytest.mark.parametrize("dim", [4, 6, 8])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_xymodel_nn_lagrangian_shift_invariance(rng, dense_layers, dim, offset):
    """Check that the neural network Lagrangian has the same value
    if all angles are shifted by a fixed offset.

    In other words, this verifies that

      L(theta_0,theta_1,...,theta_{d-1},dot(theta)_0,dot(theta)_1,...,dot(theta)_{d-1})
      = L(theta_1,...,theta_{d-1},theta_0,dot(theta)_1,...,dot(theta)_{d-1},dot(theta)_0)

    :arg dim: dimension of system
    :arg offset: shift offset
    """
    nn_lagrangian = XYModelNNLagrangian(
        dim, dense_layers, rotation_invariant=True, shift_invariant=True
    )
    # number of samples to check
    n_samples = 4
    # rotation angle
    phi = 1.1
    # tolerance for tests
    tolerance = 2.0e-5
    q = rng.uniform(low=-np.pi, high=+np.pi, size=(n_samples, dim))
    qdot = rng.normal(size=(n_samples, dim))
    X = np.concatenate([q, qdot], axis=1)
    X_shifted = np.concatenate(
        [np.roll(q, offset, axis=1), np.roll(qdot, offset, axis=1)], axis=1
    )
    dL = nn_lagrangian(X_shifted) - nn_lagrangian(X)
    assert np.linalg.norm(dL) < tolerance


@pytest.mark.skip(reason="too expensive")
def test_xymodel_nn_eigenstate_shift_invariance(rng, dense_layers):
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
        dim, dense_layers, rotation_invariant=True, shift_invariant=True
    )
    model = NNLagrangianModel(nn_lagrangian)
    time_integrator = RK4Integrator(model, dt)
    alpha_1 = 1.3
    alpha_2 = 2.4
    delta_1 = 0.3
    delta_2 = 1.7
    # tolerance for tests
    tolerance = 2.0e-5
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


@pytest.mark.parametrize("dim", [2, 4, 6, 8])
@pytest.mark.parametrize("reflection_invariant", [False, True])
def test_single_particle_lagrangian_rotation_invariance(
    rng, dense_layers, dim, reflection_invariant
):
    """Check that the neural network Lagrangian has the same value
    if the input vectors are rotated.

    :arg dim: dimension of system
    :arg reflection:invariant: test reflection invariance
    """
    nn_lagrangian = SingleParticleNNLagrangian(
        dim, dense_layers, rotation_invariant=True
    )
    # number of samples to check
    n_samples = 4
    # tolerance for tests
    tolerance = 2.0e-5
    q = rng.normal(size=(n_samples, dim))
    qdot = rng.normal(size=(n_samples, dim))
    if reflection_invariant:
        R_rot = ortho_group.rvs(dim, random_state=rng)
    else:
        R_rot = special_ortho_group.rvs(dim, random_state=rng)
    X = np.concatenate([q, qdot], axis=1)
    rotate = lambda v: np.einsum("ij,aj->ai", R_rot, v)
    X_rotated = np.concatenate([rotate(q), rotate(qdot)], axis=1)
    dL = nn_lagrangian(X_rotated) - nn_lagrangian(X)
    assert np.linalg.norm(dL) < tolerance


@pytest.mark.parametrize("dim_space", [2, 3, 4])
@pytest.mark.parametrize("rotation_invariant", [False, True])
@pytest.mark.parametrize("translation_invariant", [False, True])
@pytest.mark.parametrize("reflection_invariant", [False, True])
def test_two_particle_lagrangian_invariance(
    rng,
    dense_layers,
    dim_space,
    rotation_invariant,
    translation_invariant,
    reflection_invariant,
):
    """Check that the neural network Lagrangian has the same value
    if the input vectors are rotated or translated.

    :arg dim_space: dimension of space
    :arg rotation_invariant: test rotation invariance
    :arg translation_invariant: test translation invariance
    :arg reflection_invariant: test reflection invariance
    """
    # Reflection only makes sense if we also assume rotational invariance
    if reflection_invariant and not rotation_invariant:
        pytest.skip(
            "reflection invariance only defined in context of rotational invariance"
        )
    nn_lagrangian = TwoParticleNNLagrangian(
        dim_space,
        dense_layers,
        rotation_invariant=rotation_invariant,
        translation_invariant=translation_invariant,
        reflection_invariant=reflection_invariant,
    )
    # number of samples to check
    n_samples = 4
    # tolerance for tests
    tolerance = 2.5e-5
    x1 = rng.normal(size=(n_samples, dim_space))
    x2 = rng.normal(size=(n_samples, dim_space))
    u1 = rng.normal(size=(n_samples, dim_space))
    u2 = rng.normal(size=(n_samples, dim_space))
    if reflection_invariant:
        R_rot = ortho_group.rvs(dim_space, random_state=rng)
    else:
        R_rot = special_ortho_group.rvs(dim_space, random_state=rng)
    offset = rng.normal(size=dim_space)
    X = np.concatenate([x1, x2, u1, u2], axis=1)
    rotate = lambda v: np.einsum("ij,aj->ai", R_rot, v) if rotation_invariant else v
    translate = lambda v: v + offset if translation_invariant else v
    X_transformed = np.concatenate(
        [rotate(translate(x1)), rotate(translate(x2)), rotate(u1), rotate(u2)], axis=1
    )
    dL = nn_lagrangian(X_transformed) - nn_lagrangian(X)
    assert np.linalg.norm(dL) < tolerance


def test_schwarzschild_lagrangian_rotation_invariance(rng, dense_layers):
    """Check that the neural network Lagrangian has the same value
    if the three-vector parts of the input vectors are rotated.
    """
    nn_lagrangian = SchwarzschildNNLagrangian(dense_layers, rotation_invariant=True)
    # number of samples to check
    n_samples = 4
    # tolerance for tests
    tolerance = 2.0e-5
    q = rng.normal(size=(n_samples, 4))
    qdot = rng.normal(size=(n_samples, 4))
    R_rot = np.identity(4)
    R_rot[1:, 1:] = ortho_group.rvs(3, random_state=rng)
    X = np.concatenate([q, qdot], axis=1)
    rotate = lambda v: np.einsum("ij,aj->ai", R_rot, v)
    X_rotated = np.concatenate([rotate(q), rotate(qdot)], axis=1)
    dL = nn_lagrangian(X_rotated) - nn_lagrangian(X)
    assert np.linalg.norm(dL) < tolerance
