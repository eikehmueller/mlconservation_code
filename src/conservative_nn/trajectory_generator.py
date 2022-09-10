"""Code for generating trajectories while monitoring quantities of interest"""
import os
import numpy as np
from conservative_nn.time_integrator import RK4Integrator


class TrajectoryGenerator:
    def __init__(
        self,
        dynamical_system,
        initializer,
        monitors,
        monitor_filepath,
        dt=0.01,
        t_final=1.0,
    ):
        """Generate trajectory by numerically integrating the chosen dynamical system

        The monitored data can be stored in json files. The names of these files are given as

        monitor_filepath + "_monitor_"+str(j)+".json", where j is the index of the monitor.

        :arg dynamical_system: system to integrate
        :arg initializer: initialiser
        :arg monitors: monitors to record along trajectory
        :arg monitor_filepath: filepath for monitor files (see above)
        :arg dt: numerical timestep size
        :arg t_final: final time
        """
        self.dim = dynamical_system.dim
        self.dynamical_system = dynamical_system
        self.initializer = initializer
        self.dt = dt
        self.t_final = t_final
        self.monitors = monitors
        self.monitor_filepath = monitor_filepath
        nsteps = int(self.t_final / self.dt)
        self.t = np.zeros(nsteps + 1)
        for monitor in self.monitors:
            monitor.reset()
        self.time_integrator = RK4Integrator(self.dynamical_system, self.dt)

    def run(self):
        """Integrate the dynamical system forward in time, while monitoring quantities.

        If all monitor files exist already, the code will *not* run and the monitor data
        will be read from disk instead.
        """
        nsteps = int(self.t_final / self.dt)
        self.t = [j * self.dt for j in range(nsteps + 1)]
        try:
            self.load_monitors()
        except:
            q, qdot = self.initializer.draw()
            self.time_integrator.set_state(q, qdot)
            for j in range(nsteps + 1):
                for monitor in self.monitors:
                    monitor(self.time_integrator)
                self.time_integrator.integrate(1)
            self.save_monitors()

    def save_monitors(self):
        """Save monitor data to json file, if this files does not yet exist"""
        for j, monitor in enumerate(self.monitors):
            filepath = self.monitor_filepath + f"_monitor_{j:d}.json"
            if not os.path.exists(filepath):
                monitor.save_json(filepath)

    def load_monitors(self):
        """Load monitor data from json file"""
        for j, monitor in enumerate(self.monitors):
            filepath = self.monitor_filepath + f"_monitor_{j:d}.json"
            monitor.load_json(filepath)
