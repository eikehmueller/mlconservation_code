"""Data generator classes

These classes can be used to construct data generator for training the neural networks
"""
import numpy as np
import tensorflow as tf
from time_integrator import RK4Integrator


class DynamicalSystemDataGenerator:
    """Class for generating data for training Lagrangan Neural Network integrators

    Generates samples of the form (X^{(n)},y^{(n)}) with

      X^{(n)} = q^{(n)} + xi_q^{(n)}, qdot^{(n)} + xi_{qdot}^{(n)}

    and

      y^{(n)} = qdotdot^{(n)} + xi_{qdotdot}^{(n)}

    where the acceleration qqdot^{(n)} is computed from q^{(n)}, qdot^{(n)} with
    the dynamical system. xi^{(n)} are random vectors drawn from a normal distribution
    with a given standard deviation sigma.
    """

    def __init__(
        self,
        dynamical_system,
        initializer,
        re_initialize=False,
        sigma=0.1,
        dt=0.01,
        tinterval=0.1,
    ):
        """Construct a new instance

        :arg dynamical_system: dynamical system used for training
        :arg initializer: Class which returns a (random) state (q,qdot) in phase space
        :arg sigma: standard deviation of Gaussian noise
        :arg dt: timestep in integrator
        :arg tinterval: time interval used for sampling if re_initialise is set to False
        """
        self.dynamical_system = dynamical_system
        self.initializer = initializer
        self.re_initialize = re_initialize
        self.sigma = sigma
        self.dt = dt
        self.tinterval = tinterval

        self.dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(2 * self.dynamical_system.dim),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(shape=(self.dynamical_system.dim), dtype=tf.float32),
            ),
        )

    def _generator(self):
        """Generate a new data sample (X,y)"""
        dim = self.dynamical_system.dim
        if self.re_initialize:
            while True:
                # Draw new random initial start
                q, qdot = self.initializer.draw()
                X = np.concatenate((q, qdot), axis=0)
                y = self.dynamical_system.call(X)
                dX = self.sigma * np.random.normal(size=2 * dim)
                dy = self.sigma * np.random.normal(size=dim)
                yield (X + dX, y + dy)
        else:
            n_steps = int(self.tinterval / self.dt)
            q, qdot = self.initializer.draw()
            time_integrator = RK4Integrator(self.dynamical_system, self.dt)
            time_integrator.set_state(q, qdot)
            while True:
                time_integrator.integrate(n_steps)
                X = np.concatenate([time_integrator.q, time_integrator.qdot], axis=0)
                y = self.dynamical_system.call(X)
                dX = self.sigma * np.random.normal(size=2 * dim)
                dy = self.sigma * np.random.normal(size=dim)
                yield (X + dX, y + dy)


class KeplerDataGenerator:
    """Class for generating data for Kepler system

    Generates samples of the form (X^{(n)},y^{(n)}) with

      X^{(n)} = q^{(n)} + xi_q^{(n)}, qdot^{(n)} + xi_{qdot}^{(n)}

    and

      y^{(n)} = qdotdot^{(n)} + xi_{qdotdot}^{(n)}

    where the acceleration qqdot^{(n)} is computed from q^{(n)}, qdot^{(n)} by using
    the analytical solution of a particle moving in a 1/r central force field.

    Normally distributed random noise is added to both X and y.
    """

    def __init__(self, kepler_solution, sigma=0.1):
        """Construct a new instance

        :arg kepler_solution: analytical solution object
        :arg sigma: noise level
        """
        self.kepler_solution = kepler_solution
        self.sigma = sigma
        self.dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(
                    shape=[6],
                    dtype=tf.float32,
                ),
                tf.TensorSpec(shape=[3], dtype=tf.float32),
            ),
        )

    def _generator(self):
        """Generate a new data sample (X,y)"""
        while True:
            # Draw random angle
            phi = np.random.uniform(low=-np.pi, high=+np.pi)
            # Compute corresponding position and velocity
            q = self.kepler_solution.position(phi)
            qdot = self.kepler_solution.velocity(phi)
            X = np.concatenate([q, qdot], axis=0)
            # Compute acceleration
            y = self.kepler_solution.acceleration(phi)
            dX = self.sigma * np.random.normal(size=6)
            dy = self.sigma * np.random.normal(size=3)
            yield (X + dX, y + dy)
