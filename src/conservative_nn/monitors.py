"""Monitors for quantities of interest along trajectories"""
import numpy as np
from conservative_nn.dynamical_system import (
    SchwarzschildSystem,
    TwoParticleSystem,
    DoubleWellPotentialSystem,
    KeplerSystem,
)


class Monitor:
    def __init__(self):
        """Class for accumulating data over a trajectory

        :arg n: components of quantity to monitor
        """
        self.reset()

    def reset(self):
        """Reset monitor"""
        self._data = []

    @property
    def data(self):
        """Get data as numpy array"""
        return np.asarray(self._data).transpose()


class PositionMonitor(Monitor):
    """Monitor for current position

    Stores the current position in an array of size dim x (nsteps+1)

    :arg dim: dimension of dynamical system
    """

    def __init__(self):
        super().__init__()

    def __call__(self, time_integrator):
        """Evaluate the monitor for the current state of the time_integrator

        :arg time_integrator: time integrator to monitor
        """
        self._data.append(list(time_integrator.q[:]))


class InvariantMonitor(Monitor):
    """Monitor for invariants of dynamical system

    Stores the invariants of the dynamical system in an array of size ninvariant x (nsteps+1)

    :arg lagrangian: Lagrangian of dynamical system, usually based represented
                     by a neural network
    """

    def __init__(self, lagrangian):
        super().__init__()
        self.lagrangian = lagrangian

    def __call__(self, time_integrator):
        """Evaluate the monitor for value of the invariant of the underlying system

        :arg time_integrator: time integrator to monitor
        """
        inputs = np.concatenate([time_integrator.q, time_integrator.qdot])
        self._data.append(np.asarray(self.lagrangian.invariant(inputs)).flatten())


class SingleParticleInvariantMonitor(Monitor):
    """Monitor for angular momentum invariants of a single particle

    :arg dynamical_system: underlying dynamical system
    """

    def __init__(self, dynamical_system):
        super().__init__()
        assert isinstance(
            dynamical_system, (DoubleWellPotentialSystem, KeplerSystem)
        ), "Monitor only works for instances of SingleParticleSystem"
        self.dynamical_system = dynamical_system
        self.mass = self.dynamical_system.mass
        self.dim = dynamical_system.dim

    def __call__(self, time_integrator):
        """Evaluate the monitor for value of the invariant of the underlying system

        :arg time_integrator: time integrator to monitor
        """
        x = time_integrator.q[0 : self.dim]
        u = time_integrator.qdot[0 : self.dim]
        ell = 0
        invariants = []
        for j in range(self.dim):
            for k in range(j + 1, self.dim):
                invariants.append(self.mass * (u[j] * x[k] - u[k] * x[j]))
                ell += 1
        self._data.append(invariants)


class SchwarzschildInvariantMonitor(Monitor):
    """Monitor for angular momentum invariants of a particle moving in the Scharzschild metric

    :arg dynamical_system: underlying dynamical system
    """

    def __init__(self, dynamical_system):
        super().__init__()
        assert isinstance(
            dynamical_system, (SchwarzschildSystem)
        ), "Monitor only works for instances of SchwarzschildSystem"
        self.dynamical_system = dynamical_system
        self.dim = 4

    def __call__(self, time_integrator):
        """Evaluate the monitor for value of the invariant of the underlying system

        :arg time_integrator: time integrator to monitor
        """
        x = time_integrator.q[1 : self.dim]
        u = time_integrator.qdot[1 : self.dim]
        ell = 0
        invariants = []
        for j in range(self.dim - 1):
            for k in range(j + 1, self.dim - 1):
                invariants.append((u[j] * x[k] - u[k] * x[j]))
                ell += 1
        self._data.append(invariants)


class TwoParticleInvariantMonitor(Monitor):
    """Monitor for linear- and angular momentum invariants of two-particle system

    Stores the invariants of the dynamical system in an array of size ninvariant x (nsteps+1)

    :arg dynamical_system: underlying dynamical system
    """

    def __init__(self, dynamical_system):
        super().__init__()
        assert isinstance(
            dynamical_system, TwoParticleSystem
        ), "Monitor only works for instances of TwoParticleSystem"
        self.dynamical_system = dynamical_system
        self.mass1 = self.dynamical_system.mass1
        self.mass2 = self.dynamical_system.mass2
        self.dim = dynamical_system.dim
        self.dim_space = self.dim // 2

    def __call__(self, time_integrator):
        """Evaluate the monitor for value of the invariant of the underlying system

        :arg time_integrator: time integrator to monitor
        """
        x1 = time_integrator.q[0 : self.dim_space]
        x2 = time_integrator.q[self.dim_space : 2 * self.dim_space]
        u1 = time_integrator.qdot[0 : self.dim_space]
        u2 = time_integrator.qdot[self.dim_space : 2 * self.dim_space]
        invariants = np.zeros(self.dim * (self.dim + 1) // 2)
        # Linear momentum
        invariants[: self.dim_space] = self.mass1 * u1[:] + self.mass2 * u2[:]
        ell = self.dim_space
        for j in range(self.dim_space):
            for k in range(j + 1, self.dim_space):
                invariants[ell] = self.mass1 * (
                    u1[j] * x1[k] - u1[k] * x1[j]
                ) + self.mass2 * (u2[j] * x2[k] - u2[k] * x2[j])
                ell += 1
        self._data.append(list(invariants))


class VelocitySumMonitor(Monitor):
    """Monitor for sum of velocities

    :arg dim: dimension of dynamical system
    """

    def __init__(self):
        super().__init__()

    def __call__(self, time_integrator):
        """Evaluate the monitor for the current state of the time_integrator

        :arg time_integrator: time integrator to monitor
        """
        self._data.append(np.sum(time_integrator.qdot[:]))
