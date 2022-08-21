"""Code for generating trajectories while monitoring quantities of interest"""
import numpy as np
from conservative_nn.time_integrator import RK4Integrator


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
            monitor.reset()
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
