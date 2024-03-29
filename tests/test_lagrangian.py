import numpy as np
import pytest
import tensorflow as tf

from conservative_nn.lagrangian import (
    DoubleWellPotentialLagrangian,
    HarmonicOscillatorLagrangian,
    XYModelLagrangian,
    DoublePendulumLagrangian,
    RelativisticChargedParticleLagrangian,
    DoubleWellPotentialLagrangian,
    TwoParticleLagrangian,
    KeplerLagrangian,
    SchwarzschildLagrangian,
)
from conservative_nn.dynamical_system import (
    DoubleWellPotentialSystem,
    HarmonicOscillatorSystem,
    RelativisticChargedParticleSystem,
    XYModelSystem,
    DoublePendulumSystem,
    DoubleWellPotentialSystem,
    TwoParticleSystem,
    KeplerSystem,
    SchwarzschildSystem,
)
from conservative_nn.lagrangian_dynamical_system import (
    LagrangianDynamicalSystem,
    RelativisticChargedParticleLagrangianDynamicalSystem,
)
from common import harmonic_oscillator_matrices, rng, tolerance

"""Tests for the Lagrangian and dynamical system.

Checks that the the Lagrangians are computed correctly by comparing them
to a manual computation. In addition, the acceleration derived from the Lagrangians
(via LagrangianDynamicalSystem) is compared to the direct computation of the
acceleration with the corresponding function of the dynamical system class.
"""


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_harmonic_oscillator_lagrangian(rng, tolerance, dim):
    """Check that the Harmonic Oscillator Lagrangian is computer correctly
    by comparing to a manual calculation.

    :arg dim: dimension of state space
    """
    M_mat, A_mat = harmonic_oscillator_matrices(dim)
    lagrangian = HarmonicOscillatorLagrangian(dim, M_mat, A_mat)
    q = rng.standard_normal(size=dim)
    qdot = rng.standard_normal(size=dim)
    q_qdot = np.array(np.concatenate((q, qdot)).reshape((1, 2 * dim)), dtype=np.float64)
    L = lagrangian(q_qdot).numpy()[0]
    L_manual = 0.5 * np.dot(qdot, np.dot(M_mat, qdot)) - 0.5 * np.dot(
        q, np.dot(A_mat, q)
    )
    assert abs(L - L_manual) < tolerance


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_harmonic_oscillator_acceleration(rng, tolerance, dim):
    """Check that the acceleration is correct for the Harmonic Oscillator
    Lagrangian. Not that in this case we have that

    d^2q/dt^2 = M^{-1} A q

    Evaluate this for a random phase space vector (q,qdot).

    :arg dim: dimension of state space
    """
    M_mat, A_mat = harmonic_oscillator_matrices(dim)
    lagrangian = HarmonicOscillatorLagrangian(dim, M_mat, A_mat)
    lagrangian_dynamical_system = LagrangianDynamicalSystem(lagrangian)
    q_qdot = tf.constant(rng.standard_normal(size=(1, 2 * dim)), dtype=tf.float64)
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    dynamical_system = HarmonicOscillatorSystem(dim, M_mat, A_mat)
    acc = dynamical_system.call(np.reshape(q_qdot, (2 * dim)))
    assert np.linalg.norm(lagrangian_acc - acc) < tolerance


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_xy_model_lagrangian(rng, tolerance, dim):
    """Check that the Lagrangian of the XY model is computed correctly
    by comparing to a manual calculation.

    :arg dim: dimension of state space
    """
    lagrangian = XYModelLagrangian(dim)
    q = rng.standard_normal(size=dim)
    qdot = rng.standard_normal(size=dim)
    q_qdot = np.array(np.concatenate((q, qdot)).reshape((1, 2 * dim)), dtype=np.float64)
    L = lagrangian(q_qdot).numpy()[0]
    a_lat = 1.0 / dim
    L_manual = 0.5 * a_lat * np.dot(qdot, qdot) - 1.0 / a_lat * np.sum(
        -np.cos(q - np.roll(q, -1)) + 1.0
    )
    assert abs(L - L_manual) < tolerance


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_xy_model_acceleration(rng, tolerance, dim):
    """Check that the acceleration is correct for the XY Model
    Lagrangian.

    Evaluate this for a random phase space vector (q,qdot)

    :arg dim: dimension of state space
    """
    lagrangian = XYModelLagrangian(dim)
    lagrangian_dynamical_system = LagrangianDynamicalSystem(lagrangian)
    q_qdot = tf.constant(rng.standard_normal(size=(1, 2 * dim)), dtype=tf.float64)
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    dynamical_system = XYModelSystem(dim)
    acc = dynamical_system.call(np.reshape(q_qdot, (2 * dim)))
    assert np.linalg.norm(lagrangian_acc - acc) < tolerance


def test_double_pendulum_acceleration(rng, tolerance):
    """Check that the acceleration is correct for the double pendulum
    Lagrangian.

    Evaluate this for a random phase space vector (q,qdot)

    :arg dim: dimension of state space
    """
    m0 = 0.9
    m1 = 1.1
    L0 = 1.3
    L1 = 0.87
    lagrangian = DoublePendulumLagrangian(m0, m1, L0, L1)
    lagrangian_dynamical_system = LagrangianDynamicalSystem(lagrangian)
    q_qdot = tf.constant(rng.standard_normal(size=(1, 4)), dtype=tf.float64)
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    dynamical_system = DoublePendulumSystem(m0, m1, L0, L1)
    acc = dynamical_system.call(np.reshape(q_qdot, 4))
    assert np.linalg.norm(lagrangian_acc - acc) < tolerance


@pytest.mark.parametrize("constant_E_electric", [True, False])
def test_relativistic_charged_particle_acceleration(
    rng, tolerance, constant_E_electric
):
    """Check that the acceleration is correct for the relativistic particle
    moving in a constant electromagnetic field.

    Evaluate this for a random phase space vector (q,qdot)
    """
    E_electric = [0.8, 1.3, 0.3]
    B_magnetic = [0.3, 1.1, -0.8]
    mass = 1.2
    charge = 0.864
    dynamical_system = RelativisticChargedParticleSystem(
        mass, charge, E_electric, B_magnetic, constant_E_electric=constant_E_electric
    )
    lagrangian = RelativisticChargedParticleLagrangian(mass, charge)
    lagrangian_dynamical_system = RelativisticChargedParticleLagrangianDynamicalSystem(
        lagrangian, dynamical_system.A_vec_func
    )
    q_qdot = tf.constant(rng.standard_normal(size=[1, 8]), dtype=tf.float64)
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    acc = dynamical_system.call(np.reshape(q_qdot, [8]))
    assert np.linalg.norm(lagrangian_acc - acc) < tolerance


@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5, 6])
def test_double_well_potential_acceleration(rng, tolerance, dim):
    """Check that the acceleration is correct for the double well potential Lagrangian.

    Evaluate this for a random phase space vector (q,qdot)

    :arg dim: dimension of state space
    """
    mass = 1.2
    mu = 0.97
    kappa = 1.08
    lagrangian = DoubleWellPotentialLagrangian(dim, mass, mu, kappa)
    lagrangian_dynamical_system = LagrangianDynamicalSystem(lagrangian)
    q_qdot = tf.constant(rng.standard_normal(size=(1, 2 * dim)), dtype=tf.float64)
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    dynamical_system = DoubleWellPotentialSystem(dim, mass, mu, kappa)
    acc = dynamical_system.call(np.reshape(q_qdot, (2 * dim)))
    assert np.linalg.norm(lagrangian_acc - acc) < tolerance


@pytest.mark.parametrize("dim_space", [1, 2, 3, 4, 5, 6])
def test_two_particle_acceleration(rng, tolerance, dim_space):
    """Check that the acceleration is correct for the two particle Lagrangian.

    Evaluate this for a random phase space vector (q,qdot)

    :arg dim_space: dimension of state space
    """
    mass1 = 1.2
    mass2 = 0.98
    mu = 0.97
    kappa = 1.08
    dim = 2 * dim_space
    lagrangian = TwoParticleLagrangian(dim_space, mass1, mass2, mu, kappa)
    lagrangian_dynamical_system = LagrangianDynamicalSystem(lagrangian)
    q_qdot = tf.constant(rng.standard_normal(size=(1, 2 * dim)), dtype=tf.float64)
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    dynamical_system = TwoParticleSystem(dim_space, mass1, mass2, mu, kappa)
    acc = dynamical_system.call(np.reshape(q_qdot, (2 * dim)))
    assert np.linalg.norm(lagrangian_acc - acc) < tolerance


def test_kepler_acceleration(rng, tolerance):
    """Check that the acceleration is correct for the Kepler Lagramgian.

    Evaluate this for a random phase space vector (q,qdot)
    """
    mass = 1.2
    alpha = 0.97
    lagrangian = KeplerLagrangian(mass, alpha)
    lagrangian_dynamical_system = LagrangianDynamicalSystem(lagrangian)
    q_qdot = tf.constant(rng.standard_normal(size=(1, 6)), dtype=tf.float64)
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    dynamical_system = KeplerSystem(mass, alpha)
    acc = dynamical_system.call(np.reshape(q_qdot, [6]))
    assert np.linalg.norm(lagrangian_acc - acc) < tolerance


def test_schwarzschild_acceleration(rng, tolerance):
    """Check that the acceleration is correct for the Schwarzschild Lagramgian.

    Evaluate this for a random phase space vector (q,qdot)
    """
    r_s = 0.103
    sigma = 0.25 * r_s  # strength of perturbation
    lagrangian = SchwarzschildLagrangian(r_s)
    lagrangian_dynamical_system = LagrangianDynamicalSystem(lagrangian)
    r0 = 10 * r_s
    v0 = 1.26 * np.sqrt(0.5 * r_s / r0) / np.sqrt(1 - r_s / r0)
    x0 = np.asarray([0, r0, 0, 0]) + sigma * rng.standard_normal(4)
    u0 = np.asarray([0, 0, v0, 0]) + sigma * rng.standard_normal(4)
    # make sure that the 4-velocity has norm -1
    u0[0] = lagrangian.zero_velocity(x0[1:4], u0[1:4])
    q_qdot = tf.constant(np.concatenate([x0, u0]), shape=[1, 8], dtype=tf.float64)
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    dynamical_system = SchwarzschildSystem(r_s)
    acc = dynamical_system.call(np.reshape(q_qdot, [8]))
    assert np.linalg.norm(lagrangian_acc - acc) / np.linalg.norm(acc) < tolerance
