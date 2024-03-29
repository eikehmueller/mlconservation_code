"""Tests for the Kepler solution

Check that the analytical solution is consistent with the corresponding dynamical system
"""

import numpy as np
from common import tolerance

from conservative_nn.kepler import KeplerSolution
from conservative_nn.dynamical_system import KeplerSystem


def test_kepler_acceleration_analytical(tolerance):
    """Check that the acceleration computed with the analytical solution class
    agrees with the acceleration from the corresponding dynamical system"""
    mass = 0.983
    alpha = 1.32
    excentricity = 0.76
    angular_momentum = 1.07
    phi0 = 3.5
    phi = 1.8
    kepler_solution = KeplerSolution(
        mass=mass,
        alpha=alpha,
        excentricity=excentricity,
        angular_momentum=angular_momentum,
        phi0=phi0,
    )
    dynamical_system = KeplerSystem(mass=mass, alpha=alpha)
    q = kepler_solution.position(phi)
    qdot = kepler_solution.velocity(phi)
    q_qdot = np.concatenate([q, qdot], axis=0)
    acc_analytical = kepler_solution.acceleration(phi)
    acc = dynamical_system.call(q_qdot)
    assert np.linalg.norm(acc - acc_analytical) < tolerance
