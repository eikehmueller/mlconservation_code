"""Classes for numerical time integration of dynamical systems.

The systems are assumed to be of the form

    dq/dt    = qdot
    dqdot/dt = a(q,qdot)

where the acceleration is a(q,qdot) and both q and qdot are
d-dimensional vectors. qdot is also referred to as the velocity.

If the dynamical system contains C-code snippets for computing
the acceleration, these are compiled into a library. Otherwise,
the acceleration() method of the dynamical system class is used.
"""

from abc import ABC, abstractmethod
import string
import subprocess
import ctypes
import hashlib
import re
import os
import numpy as np


class TimeIntegrator(ABC):
    """Abstract base class for a single step traditional time integrator

    :arg lagrangian: Dynamical system to be integrated
    :arg dt: time step size
    """

    def __init__(self, dynamical_system, dt):
        self.dynamical_system = dynamical_system
        self.dt = dt
        self.q = np.zeros(dynamical_system.dim)
        self.qdot = np.zeros(dynamical_system.dim)
        self.label = None
        # Check whether dynamical system has a C-code snippet for computing the acceleration
        self.fast_code = hasattr(self.dynamical_system, "acceleration_code")

    def set_state(self, q, qdot):
        """Set the current state of the integrator to a specified
        position and velocity.

        :arg q: New position vector
        :arg qdot: New velocity vector
        """
        self.q[:] = q[:]
        self.qdot[:] = qdot[:]

    @abstractmethod
    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """

    def _generate_timestepper_library(self, c_sourcecode):
        """Generate shared library from c source code

        The generated library will implement the timestepper. Returns ctypes wrapper
        which allows calling the function

          void timestepper(double* q, double* qdot, int nsteps) { ... }

        The dynamical system class is expected to contain a code snippet for
        computing acceleration[j] given q[j] and qdot[j].

        :arg c_sourcecode: C source code
        """
        # If this is the case, auto-generate fast C code for the Velocity Verlet update

        if self.fast_code:
            if hasattr(self.dynamical_system, "preamble_code"):
                preamble = self.dynamical_system.preamble_code
            else:
                preamble = ""
            if hasattr(self.dynamical_system, "header_code"):
                header = self.dynamical_system.header_code
            else:
                header = ""
            c_substituted_sourcecode = string.Template(c_sourcecode).substitute(
                DIM=self.dynamical_system.dim,
                DT=self.dt,
                ACCELERATION_CODE=self.dynamical_system.acceleration_code,
                HEADER_CODE=header,
                PREAMBLE_CODE=preamble,
            )
            sha = hashlib.md5()
            sha.update(c_substituted_sourcecode.encode())
            directory = "./generated_code/"
            if not os.path.exists(directory):
                os.mkdir(directory)
            filestem = "./timestepper_" + sha.hexdigest()
            so_file = directory + "/" + filestem + ".so"
            source_file = directory + "/" + filestem + ".c"
            with open(source_file, "w", encoding="utf8") as f:
                print(c_substituted_sourcecode, file=f)
            # Compile source code (might have to adapt for different compiler)
            subprocess.run(
                ["gcc", "-fPIC", "-shared", "-O3", "-o", so_file, source_file],
                check=True,
            )
            timestepper_lib = ctypes.CDLL(so_file).timestepper
            # Work out the number of pointer arguments
            function_definition = list(
                filter(
                    re.compile(".*void timestepper.*").match,
                    c_substituted_sourcecode.split("\n"),
                )
            )[0]
            n_pointer_args = function_definition.count("*")
            timestepper_lib.argtypes = n_pointer_args * [
                np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ] + [
                np.ctypeslib.c_intp,
            ]
            return timestepper_lib
        return None


class ForwardEulerIntegrator(TimeIntegrator):
    """Forward Euler integrator given by

        q_j^{(t+dt)}    = q_j^{(t)}    + dt * qdot_j{(t)}
        qdot_j^{(t+dt)} = qdot_j^{(t)} + dt * a_j(q^{(t)},qdot^{(t)})

    :arg dynamical_system: Dynamical system to be integrated
    :arg dt: time step size
    """

    def __init__(self, dynamical_system, dt):

        super().__init__(dynamical_system, dt)
        self.label = "ForwardEuler"
        c_sourcecode = """
        $HEADER_CODE
        void timestepper(double* q, double* qdot, int nsteps) {
            double acceleration[$DIM];
            $PREAMBLE_CODE
            for (int k=0;k<nsteps;++k) {
                $ACCELERATION_CODE
                for (int j=0;j<$DIM;++j) {
                    q[j]    += ($DT)*qdot[j];
                    qdot[j] += ($DT)*acceleration[j];
                }
            }
        }
        """
        self.timestepper_library = self._generate_timestepper_library(c_sourcecode)

    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """
        if self.fast_code:
            self.timestepper_library(self.q, self.qdot, n_steps)
        else:
            acceleration = np.zeros(self.dynamical_system.dim)
            for _ in range(n_steps):
                # Compute acceleration
                acceleration[:] = self.dynamical_system.call(
                    np.concatenate((self.q, self.qdot))
                )
                # Update position and momentum
                self.q[:] += self.dt * self.qdot[:]
                self.qdot[:] += self.dt * acceleration[:]


class RK4Integrator(TimeIntegrator):
    """RK4 integrator given by

        (k_{1,q})_j    = qdot^{(t)}
        (k_{1,qdot})_j = a_j ( q^{(t)}, qdot^{(t)} )

        (k_{2,q})_j    = qdot^{(t)} + dt/2*k_{1,qdot} )
        (k_{2,qdot})_j = a_j ( q^{(t)} + dt/2*k_{1,q}, qdot^{(t)} + dt/2*k_{1,qdot} )

        (k_{3,q})_j    = qdot^{(t)} + dt/2*k_{2,qdot} )
        (k_{3,qdot})_j = a_j ( q^{(t)} + dt/2*k_{2,q}, qdot^{(t)} + dt/2*k_{2,qdot} )

        (k_{4,q})_j    = qdot^{(t)} + dt*k_{3,qdot} )
        (k_{4,qdot})_j = a_j ( q^{(t)} + dt*k_{3,q}, qdot^{(t)} + dt*k_{3,qdot} )

        q^{(t+dt)}    = q^{(t)} + dt/6*( k_{1,q} + 2*k_{2,q} + 2*k_{3,q} + k_{4,q} )
        qdot^{(t+dt)} = qdot^{(t)} + dt/6*( k_{1,qdot} + 2*k_{2,qdot} + 2*k_{3,qdot} + k_{4,qdot} )

    :arg dynamical_system: Dynamical system to be integrated
    :arg dt: time step size
    """

    def __init__(self, dynamical_system, dt):
        super().__init__(dynamical_system, dt)
        self.label = "RK4"
        # temporary fields
        self.k1q = np.zeros(self.dynamical_system.dim)
        self.k1qdot = np.zeros(self.dynamical_system.dim)
        self.k2q = np.zeros(self.dynamical_system.dim)
        self.k2qdot = np.zeros(self.dynamical_system.dim)
        self.k3q = np.zeros(self.dynamical_system.dim)
        self.k3qdot = np.zeros(self.dynamical_system.dim)
        self.k4q = np.zeros(self.dynamical_system.dim)
        self.k4qdot = np.zeros(self.dynamical_system.dim)
        c_sourcecode = """
            $HEADER_CODE
            void timestepper(double* q, double* qdot, int nsteps) {
                // acceleration
                double acceleration[$DIM];
                // increments k_j
                double k1q[$DIM];   
                double k1qdot[$DIM];
                double k2q[$DIM];
                double k2qdot[$DIM];
                double k3q[$DIM];
                double k3qdot[$DIM];
                double k4q[$DIM];
                double k4qdot[$DIM];
                // Fields at time t
                double qt[$DIM];    
                double qdott[$DIM];
                $PREAMBLE_CODE
                for (int k=0;k<nsteps;++k) {
                    // *** Stage 1 *** compute k1
                    $ACCELERATION_CODE
                    for (int j=0;j<$DIM;++j) {
                        qt[j] = q[j];
                        qdott[j] = qdot[j];
                        k1q[j] = qdot[j];
                        k1qdot[j] = acceleration[j];
                        q[j] += 0.5*($DT)*k1q[j];
                        qdot[j] += 0.5*($DT)*k1qdot[j];
                    }
                    // *** Stage 2 *** compute k2
                    $ACCELERATION_CODE
                    for (int j=0;j<$DIM;++j) {
                        k2q[j] = qdot[j];
                        k2qdot[j] = acceleration[j];
                        q[j] = qt[j] + 0.5*($DT)*k2q[j];
                        qdot[j] = qdott[j] + 0.5*($DT)*k2qdot[j];
                    }
                    // *** Stage 3 *** compute k3
                    $ACCELERATION_CODE
                    for (int j=0;j<$DIM;++j) {
                        k3q[j] = qdot[j];
                        k3qdot[j] = acceleration[j];
                        q[j] = qt[j] + ($DT)*k3q[j];
                        qdot[j] = qdott[j] + ($DT)*k3qdot[j];
                    }
                    // *** Stage 4 *** compute k4
                    $ACCELERATION_CODE
                    for (int j=0;j<$DIM;++j) {
                        k4q[j] = qdot[j];
                        k4qdot[j] = acceleration[j];
                    }
                    // *** Final stage *** combine k's to compute q^{(t+dt)} and p^{(t+dt)}
                    for (int j=0;j<$DIM;++j) {
                        q[j] = qt[j] + ($DT)/6.*(k1q[j]+2.*k2q[j]+2.*k3q[j]+k4q[j]);
                        qdot[j] = qdott[j] + ($DT)/6.*(k1qdot[j]+2.*k2qdot[j]+2.*k3qdot[j]+k4qdot[j]);
                    }
                }
            }
            """
        self.timestepper_library = self._generate_timestepper_library(c_sourcecode)

    def integrate(self, n_steps):
        """Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        """
        if self.fast_code:
            self.timestepper_library(self.q, self.qdot, n_steps)
        else:
            qt = np.zeros(self.dynamical_system.dim)
            qdott = np.zeros(self.dynamical_system.dim)
            acceleration = np.zeros(self.dynamical_system.dim)
            for _ in range(n_steps):
                qt[:] = self.q[:]
                qdott[:] = self.qdot[:]
                # Stage 1: compute k1
                acceleration[:] = self.dynamical_system.call(
                    np.concatenate((self.q, self.qdot))
                )
                self.k1q[:] = self.qdot[:]
                self.k1qdot[:] = acceleration[:]
                # Stage 2: compute k2
                self.q[:] += 0.5 * self.dt * self.k1q[:]
                self.qdot[:] += 0.5 * self.dt * self.k1qdot[:]
                acceleration[:] = self.dynamical_system.call(
                    np.concatenate((self.q, self.qdot))
                )
                # Stage 3: compute k3
                self.k2q[:] = self.qdot[:]
                self.k2qdot[:] = acceleration[:]
                self.q[:] = qt[:] + 0.5 * self.dt * self.k2q[:]
                self.qdot[:] = qdott[:] + 0.5 * self.dt * self.k2qdot[:]
                acceleration[:] = self.dynamical_system.call(
                    np.concatenate((self.q, self.qdot))
                )
                self.k3q[:] = self.qdot[:]
                self.k3qdot[:] = acceleration[:]
                # Stage 4: compute k4
                self.q[:] = qt[:] + self.dt * self.k3q[:]
                self.qdot[:] = qdott[:] + self.dt * self.k3qdot[:]
                acceleration[:] = self.dynamical_system.call(
                    np.concatenate((self.q, self.qdot))
                )
                self.k4q[:] = self.qdot[:]
                self.k4qdot[:] = acceleration[:]
                # Final stage: combine k's
                self.q[:] = qt[:] + self.dt / 6.0 * (
                    self.k1q[:] + 2.0 * self.k2q[:] + 2.0 * self.k3q[:] + self.k4q[:]
                )
                self.qdot[:] = qdott[:] + self.dt / 6.0 * (
                    self.k1qdot[:]
                    + 2.0 * self.k2qdot[:]
                    + 2.0 * self.k3qdot[:]
                    + self.k4qdot[:]
                )
