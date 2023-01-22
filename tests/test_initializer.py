"""Tests initializer

Check that the total momentum is zero for the two- and multi-particle initializers
"""

import numpy as np
import pytest
from common import tolerance, rng

from conservative_nn.initializer import (
    TwoParticleConstantInitializer,
    MultiParticleConstantInitializer,
)


@pytest.mark.parametrize("dim_space", [2, 3, 4, 6])
def test_two_particle_constant_initializer(tolerance, dim_space):
    """Check that the total momentum of the two-particle constant initializer is zero"""
    dim = 2 * dim_space
    mass1 = 0.89
    mass2 = 1.23
    initializer = TwoParticleConstantInitializer(dim=dim, mass1=mass1, mass2=mass2)
    _, qdot = initializer.draw(dtype=np.float64)
    u1, u2 = qdot[:dim_space], qdot[dim_space:]
    P_tot = mass1 * u1 + mass2 * u2
    assert np.linalg.norm(P_tot) < tolerance


@pytest.mark.parametrize("n_part", [2, 3, 4, 5])
@pytest.mark.parametrize("dim_space", [2, 3, 4, 6])
def test_multi_particle_constant_initializer(tolerance, rng, n_part, dim_space):
    """Check that the total momentum of the multi-particle constant initializer is zero"""
    masses = rng.uniform(low=0.4, high=1.3, size=n_part)
    initializer = MultiParticleConstantInitializer(
        n_part=n_part, dim_space=dim_space, masses=masses
    )
    _, qdot = initializer.draw(dtype=np.float64)
    u = qdot.reshape((n_part, dim_space))
    P_tot = np.zeros(dim_space)
    for j in range(n_part):
        P_tot += masses[j] * u[j]
    assert np.linalg.norm(P_tot) < tolerance
