import numpy as np
import tensorflow as tf
from dynamical_system import LagrangianDynamicalSystem


class XYModelNNLagrangian(tf.keras.models.Model):
    """Neural network representation of Lagrangian for the XY model

    Rotational invariance

        L(q_0,...,q_{d-1};qdot_0,...,qdot_{d-1})
          = L(q_0+phi,...,q_{d-1}+phi;qdot_0,...,dot_{d-1})

    is exactly enforced by taking the input

      y = (q_0,q_1,...,q_{d-1},qdot_0,qdot_1,...,qdot_{d-1}) where

    and converting it to a 2d dimensional vector

      z = (q_0-q_{d-1},q_1-q_0,...,q_{d-1}-q_{d-2},qdot_1,...,qdot_{d-1})

    :arg dim: dimension d = number of spins
    :arg rotation_invariant: enforce rotational invariance
    :arg shift_invariant: enforce shift invariance
    """

    def __init__(self, dim, rotation_invariant=True, shift_invariant=True):
        super(XYModelNNLagrangian, self).__init__()
        self.dim = dim
        self.rotation_invariant = rotation_invariant
        self.shift_invariant = shift_invariant
        self.dense_layers = [
            tf.keras.layers.Dense(8, activation="tanh"),
            tf.keras.layers.Dense(8, activation="tanh"),
            tf.keras.layers.Dense(1, use_bias=False),
        ]

    def call(self, inputs):
        """Evaluate the Lagrangian for a given vector (q,qdot)

        :arg inputs: 2d-dimensional phase space vector (q,qdot)
        """
        noffsets = self.dim if self.shift_invariant else 1
        outputs = []
        for offset in range(noffsets):
            x = inputs
            # Shift by offset
            q_qdot = tf.unstack(inputs, axis=1)
            q = tf.roll(tf.stack(q_qdot[: self.dim], axis=1), offset, axis=1)
            qdot = tf.roll(tf.stack(q_qdot[self.dim :], axis=1), offset, axis=1)
            x = tf.concat([q, qdot], axis=1)
            if self.rotation_invariant:
                # replace q_j by q_j - q_{j-1} to take rotational symmetry
                # into account
                q_qdot = tf.unstack(x, axis=1)
                q = q_qdot[: self.dim]
                qdot = q_qdot[self.dim :]
                dq = [q[k] - q[(k + self.dim - 1) % self.dim] for k in range(self.dim)]
                x = tf.stack(dq + qdot, axis=1)
            for layer in self.dense_layers:
                x = layer(x)
            outputs.append(x)
        return 1.0 / noffsets * tf.add_n(outputs)

    def get_config(self):
        """Get the model configuration"""
        return {"dim": self.dim, "rotation_invariant": self.rotation_invariant}

    @classmethod
    def from_config(cls, config):
        """Reconstruct model from configuration"""
        return cls(**config)

    @tf.function
    def invariant(self, inputs):
        """Compute the quantity that is invariant under rotations

        I(t) = sum_{j=0}^{d-1} dL/d dot(q)_j

        :arg inputs: 2d-dimensional phase space vector (q,qdot)
        """
        grad_L = tf.unstack(tf.gradients(self.call(inputs), inputs)[0], axis=1)
        return tf.reduce_sum(tf.stack(grad_L[self.dim :], axis=1), axis=1)


class LagrangianModel(tf.keras.models.Model):
    """Neural network for representing the mapping from the current
    state (q,qdot) to the acceleration, assuming that the
    Lagrangian is represented by a neural network

    Represents a function R^{2d} -> R which encodes a Lagrangian

    :arg dim: dimension d of dynamical system
    :arg rotation_invariant: is the
    """

    def __init__(self, nn_lagrangian, **kwargs):
        super(LagrangianModel, self).__init__(**kwargs)
        self.dynamical_system = LagrangianDynamicalSystem(nn_lagrangian)

    def call(self, inputs):
        """Evaluate Dynamical system defined by the model Lagrangian
        for a given set of inputs

        :arg inputs: phase space vector q,qdot
        """
        return self.dynamical_system(inputs)
