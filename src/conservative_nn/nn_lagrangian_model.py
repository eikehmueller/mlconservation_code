"""Wrapper for Lagrangian neural networks

The NNLagrangianModel class wraps the Lagrangian dynamical system from
lagrangian_dynamical_system.py as a subclass of tf.keras.models.Model. This 
allows using NNLagrangianModel just like a dynamical system in time integrators
"""

import tensorflow as tf
from conservative_nn.lagrangian_dynamical_system import LagrangianDynamicalSystem


class NNLagrangianModel(tf.keras.models.Model):
    """Neural network for representing the mapping from the current
    state (q,qdot) to the acceleration, assuming that the
    Lagrangian is represented by a neural network

    Represents a function R^{2d} -> R which encodes a Lagrangian

    :arg dim: dimension d of dynamical system
    :arg rotation_invariant: is the
    """

    def __init__(self, nn_lagrangian, **kwargs):
        super().__init__(**kwargs)
        self.dynamical_system = LagrangianDynamicalSystem(nn_lagrangian)
        self.dim = self.dynamical_system.dim

    def call(self, inputs):
        """Evaluate Dynamical system defined by the model Lagrangian
        for a given set of inputs

        :arg inputs: phase space vector q,qdot
        """
        if len(inputs.shape) == 1:
            # If input is a one-dimensional tensor, extend to two dimensions to be able to use
            # the tensorflow code which expects tensors of shape (batchsize,dim)
            return (
                self.dynamical_system(tf.constant(inputs, shape=(1, inputs.shape[0])))
                .numpy()
                .flatten()
            )
        return self.dynamical_system(inputs)
