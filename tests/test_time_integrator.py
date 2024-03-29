"""Test numerical time integrators"""
import numpy as np
import pytest

from conservative_nn.dynamical_system import (
    HarmonicOscillatorSystem,
    XYModelSystem,
    DoublePendulumSystem,
    RelativisticChargedParticleSystem,
    DoubleWellPotentialSystem,
    TwoParticleSystem,
    KeplerSystem,
    SchwarzschildSystem,
)
from conservative_nn.time_integrator import ForwardEulerIntegrator, RK4Integrator
from common import harmonic_oscillator_matrices, rng, tolerance


@pytest.fixture
def harmonic_oscillator_system():
    """Construct HarmonicOscillator dynamical system object"""
    dim = 4
    M_mat, A_mat = harmonic_oscillator_matrices(dim)
    return HarmonicOscillatorSystem(dim, M_mat, A_mat)


@pytest.fixture
def xy_model_system():
    """Construct XY model dynamical system object"""
    dim = 4
    return XYModelSystem(dim)


@pytest.fixture
def double_pendulum_system():
    """Construct double pendulum dynamical system object"""
    m0 = 0.9
    m1 = 1.1
    L0 = 1.3
    L1 = 0.87
    return DoublePendulumSystem(m0, m1, L0, L1)


@pytest.fixture
def relativistic_charged_particle_system_varying_E():
    """Construct relativistic charged particle system object"""
    mass = 0.87
    charge = 1.1
    return RelativisticChargedParticleSystem(mass, charge, constant_E_electric=False)


@pytest.fixture
def relativistic_charged_particle_system_constant_E():
    """Construct relativistic charged particle system object"""
    mass = 0.87
    charge = 1.1
    return RelativisticChargedParticleSystem(mass, charge, constant_E_electric=True)


@pytest.fixture
def double_well_potential_system():
    """Construct double well potential system object"""
    dim = 5
    mass = 0.87
    mu = 1.1
    kappa = 0.97
    return DoubleWellPotentialSystem(dim, mass, mu, kappa)


@pytest.fixture
def two_particle_system():
    """Construct two particle system object"""
    dim = 5
    mass1 = 0.87
    mass2 = 1.03
    mu = 1.1
    kappa = 0.97
    return TwoParticleSystem(dim, mass1, mass2, mu, kappa)


@pytest.fixture
def kepler_system():
    """Construct Kepler system object"""
    mass = 0.87
    alpha = 1.07
    return KeplerSystem(mass, alpha)


@pytest.fixture
def schwarzschild_system():
    """Construct Schwarzschild system object"""
    r_s = 1.17
    return SchwarzschildSystem(r_s)


@pytest.fixture(
    params=[
        "harmonic_oscillator_system",
        "xy_model_system",
        "double_pendulum_system",
        "relativistic_charged_particle_system_constant_E",
        "relativistic_charged_particle_system_varying_E",
        "double_well_potential_system",
        "two_particle_system",
        "kepler_system",
        "schwarzschild_system",
    ]
)
def dynamical_system(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("TimeIntegratorCls", [ForwardEulerIntegrator, RK4Integrator])
def test_time_integrator(rng, tolerance, TimeIntegratorCls, dynamical_system):
    """Check that the time integrators gives the same results
    if the generated C-code is used."""
    dt = 0.1  # timestep size
    nsteps = 10  # number of timesteps
    dim = dynamical_system.dim
    q0 = rng.standard_normal(size=dim)
    qdot0 = rng.standard_normal(size=dim)
    q_c = rng.standard_normal(size=dim)
    qdot_c = rng.standard_normal(size=dim)
    time_integrator = TimeIntegratorCls(dynamical_system, dt)
    # Integrate using Python
    time_integrator.set_state(q0, qdot0)
    time_integrator.fast_code = False
    time_integrator.integrate(nsteps)
    q_python = np.array(time_integrator.q)
    qdot_python = np.array(time_integrator.qdot)
    # Integrate using C library
    time_integrator.set_state(q0, qdot0)
    time_integrator.fast_code = True
    time_integrator.integrate(nsteps)
    q_c = np.array(time_integrator.q)
    qdot_c = np.array(time_integrator.qdot)
    # Compare
    assert (np.linalg.norm(q_c - q_python) < tolerance) and (
        np.linalg.norm(qdot_c - qdot_python) < tolerance
    )
