import numpy as np
import tensorflow as tf


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

    def __init__(self, dynamical_system, initializer, sigma=0.1):
        """Construct a new instance

        :arg dynamical_system: dynamical system used for training
        :arg initializer: Class which returns a (random) state (q,qdot) in phase space
        :arg sigma: standard deviation of Gaussian noise
        """
        self.dynamical_system = dynamical_system
        self.initializer = initializer
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
        while True:
            # Draw new random initial start
            q, qdot = self.initializer.draw()
            X = np.concatenate((q, qdot), axis=0) + self.sigma * np.random.normal(
                size=2 * dim
            )
            y = self.dynamical_system.call(X) + self.sigma * np.random.normal(size=dim)
            yield (X, y)
