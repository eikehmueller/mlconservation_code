import numpy as np
import pytest
import tensorflow as tf
from lagrangian import HarmonicOscillatorLagrangian, XYModelLagrangian
from dynamical_system import (
    LagrangianDynamicalSystem,
    HarmonicOscillatorSystem,
    XYModelSystem,
)
from common import harmonic_oscillator_matrices

"""Tests for the Lagrangian and dynamical system.

Checks that the the Lagrangians are computed correctly by comparing them
to a manual computation. In addition, the acceleration derived from the Lagrangians
(via LagrangianDynamicalSystem) is compared to the direct computation of the
acceleration with the corresponding function of the dynamical system class.
"""


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_harmonic_oscillator_lagrangian(dim):
    """Check that the Harmonic Oscillator Lagrangian is computer correctly
    by comparing to a manual calculation.

    :arg dim: dimension of state space
    """
    M_mat, A_mat = harmonic_oscillator_matrices(dim)
    lagrangian = HarmonicOscillatorLagrangian(dim, M_mat, A_mat)
    q = np.random.normal(size=dim)
    qdot = np.random.normal(size=dim)
    q_qdot = np.array(np.concatenate((q, qdot)).reshape((1, 2 * dim)), dtype=np.float32)
    L = lagrangian(q_qdot).numpy()[0]
    L_manual = 0.5 * np.dot(qdot, np.dot(M_mat, qdot)) - 0.5 * np.dot(
        q, np.dot(A_mat, q)
    )
    tolerance = 1.0e-5
    assert abs(L - L_manual) < tolerance


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_harmonic_oscillator_force(dim):
    """Check that the force is correct for the Harmonic Oscillator
    Lagrangian. Not that in this case we have that

    d^2q/dt^2 = M^{-1} A q

    Evaluate this for a random phase space vector (q,qdot).

    :arg dim: dimension of state space
    """
    M_mat, A_mat = harmonic_oscillator_matrices(dim)
    lagrangian = HarmonicOscillatorLagrangian(dim, M_mat, A_mat)
    lagrangian_dynamical_system = LagrangianDynamicalSystem(lagrangian)
    q = np.random.normal(size=dim)
    qdot = np.array(np.random.normal(size=dim), dtype=np.float32)
    q_qdot = tf.constant(
        np.concatenate((q, qdot)).reshape((1, 2 * dim)), dtype=tf.float32
    )
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    dynamical_system = HarmonicOscillatorSystem(dim, M_mat, A_mat)
    acc = dynamical_system.call(np.reshape(q_qdot, (2 * dim)))
    tolerance = 1.0e-5
    assert np.linalg.norm(lagrangian_acc - acc) < tolerance


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_xy_model_lagrangian(dim):
    """Check that the Lagrangian of the XY model is computed correctly
    by comparing to a manual calculation.

    :arg dim: dimension of state space
    """
    lagrangian = XYModelLagrangian(dim)
    q = np.random.normal(size=dim)
    qdot = np.random.normal(size=dim)
    q_qdot = np.array(np.concatenate((q, qdot)).reshape((1, 2 * dim)), dtype=np.float32)
    L = lagrangian(q_qdot).numpy()[0]
    a_lat = 1.0 / dim
    L_manual = 0.5 * a_lat * np.dot(qdot, qdot) - 1.0 / a_lat * np.sum(
        -np.cos(q - np.roll(q, -1)) + 1.0
    )
    tolerance = 1.0e-5
    assert abs(L - L_manual) < tolerance


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_xy_model_force(dim):
    """Check that the force is correct for the XY Model
    Lagrangian.

    Evaluate this for a random phase space vector (q,qdot)

    :arg dim: dimension of state space
    """
    lagrangian = XYModelLagrangian(dim)
    lagrangian_dynamical_system = LagrangianDynamicalSystem(lagrangian)
    q = np.random.normal(size=dim)
    qdot = np.array(np.random.normal(size=dim), dtype=np.float32)
    q_qdot = tf.constant(
        np.concatenate((q, qdot)).reshape((1, 2 * dim)), dtype=tf.float32
    )
    lagrangian_acc = lagrangian_dynamical_system.call(q_qdot)
    dynamical_system = XYModelSystem(dim)
    acc = dynamical_system.call(np.reshape(q_qdot, (2 * dim)))
    tolerance = 1.0e-5
    assert np.linalg.norm(lagrangian_acc - acc) < tolerance
