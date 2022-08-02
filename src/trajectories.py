"""Code for generating trajectories while monitoring quantities of interest"""
from time_integrator import RK4Integrator
import numpy as np


class Monitor(object):
    def __init__(self, dim):
        """Class for accumulating data over a trajectory

        :arg dim: phase space dimension of dynamical system
        """
        self.dim = dim
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
        super(PositionMonitor, self).__init__(dim)

    def reset(self, nsteps):
        """reset the monitor to start new accumulation

        :arg nsteps: number of steps
        """
        super(PositionMonitor, self).reset(nsteps)
        self.q_all = np.zeros((self.dim, nsteps + 1))

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


class VelocitySumMonitor(Monitor):
    """Monitor for sum of velocities

    :arg dim: dimension of dynamical system
    """

    def __init__(self, dim):
        super(VelocitySumMonitor, self).__init__(dim)

    @property
    def value(self):
        """Return monitored quantity"""
        return self.sum_qdot

    def reset(self, nsteps):
        """reset the monitor to start new accumulation

        :arg nsteps: number of steps
        """
        super(VelocitySumMonitor, self).reset(nsteps)
        self.sum_qdot = np.zeros(nsteps + 1)

    def __call__(self, time_integrator):
        """Evaluate the monitor for the current state of the time_integrator

        :arg time_integrator: time integrator to monitor
        """
        self.sum_qdot[self.j_step] = np.sum(time_integrator.qdot[:])
        self.j_step += 1


class TrajectoryGenerator(object):
    def __init__(self, dynamical_system, initializer, monitors, dt=0.01, t_final=1.0):
        """Generate trajectory by numerically integrating the chosen dynamical system

        :arg dynamical_system: system to integrate
        :arg initializer: initialiser object
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
        t = np.zeros(nsteps + 1)
        self.time_integrator.set_state(q, qdot)
        for j in range(nsteps + 1):
            for monitor in self.monitors:
                monitor(self.time_integrator)
            self.t[j] = j * self.dt
            self.time_integrator.integrate(1)
