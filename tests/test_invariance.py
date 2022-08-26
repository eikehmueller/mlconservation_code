"""Check that the invariants are really conserved when integrating with the
trained neural network Lagrangian"""

import os
import toml
import numpy as np
import tensorflow as tf

from conservative_nn.nn_lagrangian import (
    SingleParticleNNLagrangian,
    TwoParticleNNLagrangian,
)
from conservative_nn.nn_lagrangian_model import NNLagrangianModel
from conservative_nn.initializer import (
    KeplerInitializer,
    TwoParticleConstantInitializer,
)
from conservative_nn.kepler import KeplerSolution
from conservative_nn.time_integrator import RK4Integrator


def test_invariance_kepler():
    """Check that the NN angular momentum is conserved to machine precision

    Integrate the Kepler system for a small number of timesteps and verify that
    this leaves the angular momentum (as defined for the neural network) unchanged
    """
    tolerance = 1.0e-12
    dim = 3  # dimension of problem
    dt = 0.001  # timestep size
    n_steps = 1000  # number of integration steps

    model_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../assets/trained_models/kepler/nn_lagrangian_rot",
    )
    with open(
        os.path.join(model_path, "training_parameters.toml"), "r", encoding="utf8"
    ) as toml_data:
        parameters = toml.load(toml_data)

    # parameters of Kepler system
    mass = parameters["system_specific"]["kepler"]["mass"]
    alpha = parameters["system_specific"]["kepler"]["alpha"]
    excentricity = parameters["system_specific"]["kepler"]["excentricity"]
    angular_momentum = parameters["system_specific"]["kepler"]["angular_momentum"]

    kepler_solution = KeplerSolution(
        mass=mass,
        alpha=alpha,
        excentricity=excentricity,
        angular_momentum=angular_momentum,
    )

    # Load model weights at double precision
    nn_lagrangian = SingleParticleNNLagrangian.from_saved_model(
        model_path, dtype=np.float64
    )
    dynamical_system = NNLagrangianModel(nn_lagrangian)
    time_integrator = RK4Integrator(dynamical_system, dt=dt)
    initializer = KeplerInitializer(kepler_solution)

    def J():
        """Helper function for computing the angular momentum"""
        q_qdot = np.concatenate([time_integrator.q, time_integrator.qdot])
        return np.asarray(nn_lagrangian.invariant(q_qdot)).flatten()

    q, qdot = initializer.draw()
    time_integrator.set_state(q, qdot)

    J_before = J()
    time_integrator.integrate(n_steps)
    J_after = J()
    assert np.linalg.norm(J_before - J_after) / np.linalg.norm(J_before) < tolerance


def test_invariance_two_particle():
    """Check that the NN linear- and angular momentum is conserved to machine precision

    Integrate the two particle system for a small number of timesteps and verify that
    this leaves both the linear- and angular momentum (as defined for the neural network)
    unchanged
    """
    tolerance = 1.0e-11
    dim_space = 4  # dimension of problem
    dim = 2 * dim_space
    dt = 0.001  # timestep size
    n_steps = 1000  # number of integration steps

    model_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        f"../assets/trained_models/two_particle/{dim_space:d}d/nn_lagrangian_rot_trans",
    )
    with open(
        os.path.join(model_path, "training_parameters.toml"), "r", encoding="utf8"
    ) as toml_data:
        parameters = toml.load(toml_data)

    # parameters of Kepler system
    mass1, mass2 = parameters["system_specific"]["two_particle"]["mass"]
    mu = parameters["system_specific"]["two_particle"]["mu"]
    kappa = parameters["system_specific"]["two_particle"]["kappa"]

    # Load model weights at double precision
    nn_lagrangian = TwoParticleNNLagrangian.from_saved_model(
        model_path, dtype=np.float64
    )
    dynamical_system = NNLagrangianModel(nn_lagrangian)
    time_integrator = RK4Integrator(dynamical_system, dt=dt)
    initializer = TwoParticleConstantInitializer(dim, mass1=mass1, mass2=mass2)

    def J():
        """Helper function for computing a vector with all invariants"""
        q_qdot = np.concatenate([time_integrator.q, time_integrator.qdot])
        return np.asarray(nn_lagrangian.invariant(q_qdot)).flatten()

    q, qdot = initializer.draw()
    time_integrator.set_state(q, qdot)

    J_before = J()
    time_integrator.integrate(n_steps)
    J_after = J()
    assert np.linalg.norm(J_before - J_after) / np.linalg.norm(J_before) < tolerance
