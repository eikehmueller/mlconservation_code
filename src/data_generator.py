import numpy as np
import tensorflow as tf
from time_integrator import RK4Integrator


class DataGenerator(object):
    """Class for generating data for training Lagrangan Neural Network integrators

    Generates samples of the form (X^{(n)},y^{(n)}) with

      X^{(n)} = q^{(n)} + xi_q^{(n)}, qdot^{(n)} + xi_{qdot}^{(n)}

    and

      y^{(n)} = qdotdot^{(n)} + xi_{qdotdot}^{(n)}

    where the acceleration qqdot^{(n)} is computed from q^{(n)}, qdot^{(n)} with
    the dynamical system. xi^{(n)} are random vectors drawn from a normal distribution
    with a given standard deviation sigma.
    """

    def __init__(self, dynamical_system, initializer, re_initialize=True, sigma=0.1):
        """Construct a new instance

        :arg dynamical_system: dynamical system used for training
        :arg initializer: Class which returns a (random) state (q,qdot) in phase space
        :arg sigma: standard deviation of Gaussian noise
        """
        self.dynamical_system = dynamical_system
        self.initializer = initializer
        self.re_initialize = re_initialize
        self.sigma = sigma
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
            dt = 0.02
            Tinterval = 2.0
            n_steps = int(Tinterval / dt)
            q, qdot = self.initializer.draw()
            time_integrator = RK4Integrator(self.dynamical_system, dt)
            time_integrator.set_state(q, qdot)
            while True:
                time_integrator.integrate(n_steps)
                X = np.concatenate([time_integrator.q, time_integrator.qdot], axis=0)
                y = self.dynamical_system.call(X)
                dX = self.sigma * np.random.normal(size=2 * dim)
                dy = self.sigma * np.random.normal(size=dim)
                yield (X + dX, y + dy)
