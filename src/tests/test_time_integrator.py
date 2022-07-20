"""Test numerical time integrators"""
import numpy as np
import pytest
from dynamical_system import HarmonicOscillatorSystem, XYModelSystem
from common import harmonic_oscillator_matrices
from time_integrator import ForwardEulerIntegrator, RK4Integrator


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


@pytest.fixture(params=["harmonic_oscillator_system", "xy_model_system"])
def dynamical_system(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("TimeIntegratorCls", [ForwardEulerIntegrator, RK4Integrator])
def test_time_integrator_euler(TimeIntegratorCls, dynamical_system):
    """Check that the time integrators gives the same results
    if the generated C-code is used."""
    dt = 0.1  # timestep size
    nsteps = 10  # number of timesteps
    dim = dynamical_system.dim
    q0 = np.random.normal(size=dim)
    qdot0 = np.random.normal(size=dim)
    q_c = np.random.normal(size=dim)
    qdot_c = np.random.normal(size=dim)
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
    tolerance = 1.0e-5
    assert (np.linalg.norm(q_c - q_python) < tolerance) and (
        np.linalg.norm(qdot_c - qdot_python) < tolerance
    )
