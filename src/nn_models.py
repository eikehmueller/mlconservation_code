import numpy as np
import tensorflow as tf
from dynamical_system import LagrangianDynamicalSystem


class RotationInvariantLayer(tf.keras.layers.Layer):
    """Layer which is invariant under shifts in the position

    The input to this layer is a 2d dimensional vector

      y = (q_0,q_1,...,q_{d-1},qdot_0,qdot_1,...,qdot_{d-1}) where

    and returns a 2d dimensional vector

      z = (q_0-q_{d-1},q_1-q_0,...,q_{d-1}-q_{d-2},qdot_1,...,qdot_{d-1})

    Hence, using this layer as the first layer of a neural network will
    make sure that the neural network can only represent functions which
    depend on the difference q_j-q_{j-1}.

    :arg dim: dimension d of state vector
    """

    def __init__(self, dim, **kwargs):
        super(RotationInvariantLayer, self).__init__(**kwargs)
        self.dim = dim

    def call(self, inputs):
        """Evaluate layer for given inputs

        :arg inputs: 2d-dimensional phase space vector q,qdot
        """
        q_qdot = tf.unstack(inputs, axis=1)
        q = q_qdot[: self.dim]
        qdot = q_qdot[self.dim :]
        dq = [q[k] - q[(k + self.dim - 1) % self.dim] for k in range(self.dim)]
        return tf.stack(dq + qdot, axis=1)


class LagrangianModel(tf.keras.models.Model):
    """Neural network for representing the mapping from the current
    state (q,qdot) to the acceleration, assuming that the
    Lagrangian is represented by a neural network

    Represents a function R^{2d} -> R which encodes a Lagrangian

    :arg dim: dimension d of dynamical system
    """

    def __init__(self, dim, **kwargs):
        super(LagrangianModel, self).__init__(**kwargs)
        self.dim = dim

        # Construct Lagrangian
        lagrangian = tf.keras.models.Sequential()
        lagrangian.dim = dim
        lagrangian.add(tf.keras.Input(shape=(2 * self.dim,)))
        lagrangian.add(RotationInvariantLayer(self.dim))
        lagrangian.add(tf.keras.layers.Dense(32, activation="tanh"))
        lagrangian.add(tf.keras.layers.Dense(32, activation="tanh"))
        lagrangian.add(tf.keras.layers.Dense(1, use_bias=False))

        # Construct dynamical system that uses this Lagrangian
        self.dynamical_system = LagrangianDynamicalSystem(lagrangian)

    def call(self, inputs):
        """Evaluate Dynamical system defined by the model Lagrangian
        for a given set of inputs

        :arg inputs: phase space vector q,qdot
        """
        return self.dynamical_system.call(inputs)
