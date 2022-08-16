"""Code for generating trajectories while monitoring quantities of interest"""
import numpy as np
from conservative_nn.time_integrator import RK4Integrator
from conservative_nn.dynamical_system import (
    TwoParticleSystem,
    DoubleWellPotentialSystem,
    KeplerSystem,
)


class Monitor:
    def __init__(self, ncomp):
        """Class for accumulating data over a trajectory

        :arg n: components of quantity to monitor
        """
        self.ncomp = ncomp
        self.reset(0)

    def reset(self, nsteps):
        """Reset state of monitor to start new accumulation"""
        self.j_step = 0


class PositionMonitor(Monitor):
    """Monitor for current position

    Stores the current position in an array of size dim x (nsteps+1)

    :arg dim: dimension of dynamical system
    """

    def __init__(self, dim):
        super().__init__(dim)

    def reset(self, nsteps):
        """reset the monitor to start new accumulation

        :arg nsteps: number of steps
        """
        super().reset(nsteps)
        self.q_all = np.zeros((self.ncomp, nsteps + 1))

    @property
    def value(self):
        """Return monitored quantity"""
        return self.q_all

    def __call__(self, time_integrator):
        """Evaluate the monitor for the current state of the time_integrator

        :arg time_integrator: time integrator to monitor
        """
        self.q_all[:, self.j_step] = time_integrator.q[:]
        self.j_step += 1


class InvariantMonitor(Monitor):
    """Monitor for invariants of dynamical system

    Stores the invariants of the dynamical system in an array of size ninvariant x (nsteps+1)

    :arg lagrangian: Lagrangian of dynamical system, usually based represented
                     by a neural network
    """

    def __init__(self, lagrangian):
        self.lagrangian = lagrangian
        super().__init__(self.lagrangian.ninvariant)

    def reset(self, nsteps):
        """reset the monitor to start new accumulation

        :arg nsteps: number of steps
        """
        super().reset(nsteps)
        self.invariants = np.zeros((self.ncomp, nsteps + 1))

    @property
    def value(self):
        """Return monitored quantity"""
        return self.invariants

    def __call__(self, time_integrator):
        """Evaluate the monitor for value of the invariant of the underlying system

        :arg time_integrator: time integrator to monitor
        """
        inputs = np.concatenate([time_integrator.q, time_integrator.qdot])
        self.invariants[:, self.j_step] = np.asarray(
            self.lagrangian.invariant(inputs)
        ).flatten()
        self.j_step += 1


class SingleParticleInvariantMonitor(Monitor):
    """Monitor for angular momentum invariants of a single particle

    Stores the invariants of the dynamical system in an array of size ninvariant x (nsteps+1)

    :arg dynamical_system: underlying dynamical system
    """

    def __init__(self, dynamical_system):
        assert isinstance(
            dynamical_system, (DoubleWellPotentialSystem, KeplerSystem)
        ), "Monitor only works for instances of SingleParticleSystem"
        self.dynamical_system = dynamical_system
        self.mass = self.dynamical_system.mass
        self.dim = dynamical_system.dim
        self.ninvariant = self.dim * (self.dim - 1) // 2
        super().__init__(self.ninvariant)

    def reset(self, nsteps):
        """reset the monitor to start new accumulation

        :arg nsteps: number of steps
        """
        super().reset(nsteps)
        self.invariants = np.zeros((self.ninvariant, nsteps + 1))

    @property
    def value(self):
        """Return monitored quantity"""
        return self.invariants

    def __call__(self, time_integrator):
        """Evaluate the monitor for value of the invariant of the underlying system

        :arg time_integrator: time integrator to monitor
        """
        x = time_integrator.q[0 : self.dim]
        u = time_integrator.qdot[0 : self.dim]
        ell = 0
        for j in range(self.dim):
            for k in range(j + 1, self.dim):
                self.invariants[ell, self.j_step] = self.mass * (
                    u[j] * x[k] - u[k] * x[j]
                )
                ell += 1
        self.j_step += 1


class TwoParticleInvariantMonitor(Monitor):
    """Monitor for linear- and angular momentum invariants of two-particle system

    Stores the invariants of the dynamical system in an array of size ninvariant x (nsteps+1)

    :arg dynamical_system: underlying dynamical system
    """

    def __init__(self, dynamical_system):
        assert isinstance(
            dynamical_system, TwoParticleSystem
        ), "Monitor only works for instances of TwoParticleSystem"
        self.dynamical_system = dynamical_system
        self.mass1 = self.dynamical_system.mass1
        self.mass2 = self.dynamical_system.mass2
        self.dim = dynamical_system.dim
        self.dim_space = self.dim // 2
        self.ninvariant = self.dim_space * (self.dim_space + 1) // 2
        super().__init__(self.ninvariant)

    def reset(self, nsteps):
        """reset the monitor to start new accumulation

        :arg nsteps: number of steps
        """
        super().reset(nsteps)
        self.invariants = np.zeros((self.ninvariant, nsteps + 1))

    @property
    def value(self):
        """Return monitored quantity"""
        return self.invariants

    def __call__(self, time_integrator):
        """Evaluate the monitor for value of the invariant of the underlying system

        :arg time_integrator: time integrator to monitor
        """
        x1 = time_integrator.q[0 : self.dim_space]
        x2 = time_integrator.q[self.dim_space : 2 * self.dim_space]
        u1 = time_integrator.qdot[0 : self.dim_space]
        u2 = time_integrator.qdot[self.dim_space : 2 * self.dim_space]
        # Linear momentum
        self.invariants[: self.dim_space, self.j_step] = (
            self.mass1 * u1[:] + self.mass2 * u2[:]
        )
        ell = self.dim_space
        for j in range(self.dim_space):
            for k in range(j + 1, self.dim_space):
                self.invariants[ell, self.j_step] = self.mass1 * (
                    u1[j] * x1[k] - u1[k] * x1[j]
                ) + self.mass2 * (u2[j] * x2[k] - u2[k] * x2[j])
                ell += 1
        self.j_step += 1


class VelocitySumMonitor(Monitor):
    """Monitor for sum of velocities

    :arg dim: dimension of dynamical system
    """

    def __init__(self):
        super().__init__(1)

    @property
    def value(self):
        """Return monitored quantity"""
        return self.sum_qdot

    def reset(self, nsteps):
        """reset the monitor to start new accumulation

        :arg nsteps: number of steps
        """
        super().reset(nsteps)
        self.sum_qdot = np.zeros(nsteps + 1)

    def __call__(self, time_integrator):
        """Evaluate the monitor for the current state of the time_integrator

        :arg time_integrator: time integrator to monitor
        """
        self.sum_qdot[self.j_step] = np.sum(time_integrator.qdot[:])
        self.j_step += 1


class TrajectoryGenerator:
    def __init__(self, dynamical_system, initializer, monitors, dt=0.01, t_final=1.0):
        """Generate trajectory by numerically integrating the chosen dynamical system

        :arg dynamical_system: system to integrate
        :arg initializer: initialiser
        :arg dt: numerical timestep size
        :arg t_final: final time
        """
        self.dim = dynamical_system.dim
        self.dynamical_system = dynamical_system
        self.initializer = initializer
        self.dt = dt
        self.t_final = t_final
        self.monitors = monitors
        nsteps = int(self.t_final / self.dt)
        self.t = np.zeros(nsteps + 1)
        for monitor in self.monitors:
            monitor.reset(int(self.t_final / self.dt))
        self.time_integrator = RK4Integrator(self.dynamical_system, self.dt)

    def run(self):
        """Integrate the dynamical system forward in time, while monitoring quantities"""
        q, qdot = self.initializer.draw()
        nsteps = int(self.t_final / self.dt)
        self.time_integrator.set_state(q, qdot)
        for j in range(nsteps + 1):
            for monitor in self.monitors:
                monitor(self.time_integrator)
            self.t[j] = j * self.dt
            self.time_integrator.integrate(1)
