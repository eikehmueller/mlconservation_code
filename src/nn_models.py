from itertools import combinations_with_replacement
import os
import errno
import json
import numpy as np
import tensorflow as tf
from auxilliary import ndarrayDecoder, ndarrayEncoder
from lagrangian_dynamical_system import LagrangianDynamicalSystem


class NNLagrangian(tf.keras.layers.Layer):
    """Base class for neural network based Lagrangians"""

    def __init__(self, **kwargs):
        super(NNLagrangian, self).__init__(**kwargs)
        self.dense_layers = []

    def save(self, filepath, overwrite=True):
        """Save the model to the specified directory

        :arg filepath: directory in which the model will be
        """
        # Create directory if it does not already exist
        try:
            os.makedirs(filepath)
        except OSError as e:
            if e.errno != errno.EEXIST or (not overwrite):
                raise
        # Save configuration
        with open(os.path.join(filepath, "config.json"), "w", encoding="utf8") as f:
            json.dump(self.get_config(), f, indent=4, ensure_ascii=True)

        layers = []
        for layer in self.dense_layers:
            layers.append(
                {
                    "class": type(layer).__module__ + "." + type(layer).__name__,
                    "config": layer.get_config(),
                    "weights": layer.get_weights(),
                }
            )
        with open(os.path.join(filepath, "weights.json"), "w", encoding="utf8") as f:
            json.dump(layers, f, cls=ndarrayEncoder, indent=4)

    @classmethod
    def from_saved_model(cls, filepath):
        """Construct object from saved model in a specified directory

        :arg directory: directory in which the model will be stored
        """
        import keras  # pylint: disable=reimported,redefined-outer-name,unused-import,import-outside-toplevel

        # Load configurstion
        with open(os.path.join(filepath, "config.json"), "r", encoding="utf8") as f:
            config = json.load(f)

        # Construct new instance of model
        model = cls.from_config(config)
        # Set layers and layer weights
        model.dense_layers = []
        layer_weights = {}
        with open(os.path.join(filepath, "weights.json"), "r", encoding="utf8") as f:
            layer_list = json.load(f, cls=ndarrayDecoder)
        for layer_spec in layer_list:
            layer_cls = eval(layer_spec["class"])
            config = layer_spec["config"]
            weights = layer_spec["weights"]
            layer = layer_cls.from_config(config)
            layer_weights[layer.name] = weights
            model.dense_layers.append(layer)
        # Call model once to initialise layer shapes
        inputs = tf.constant(np.zeros([1, 2 * model.dim]))
        outputs = model(inputs)
        # Now set the layer weights
        for layer in model.dense_layers:
            layer.set_weights(layer_weights[layer.name])
        return model


class XYModelNNLagrangian(NNLagrangian):
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

    def __init__(self, dim, rotation_invariant=True, shift_invariant=True, **kwargs):
        super(XYModelNNLagrangian, self).__init__(**kwargs)
        self.dim = dim
        self.rotation_invariant = rotation_invariant
        self.shift_invariant = shift_invariant
        self.dense_layers = [
            tf.keras.layers.Dense(4, activation="tanh"),
            tf.keras.layers.Dense(4, activation="tanh"),
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
        return {
            "dim": self.dim,
            "rotation_invariant": self.rotation_invariant,
            "shift_invariant": self.shift_invariant,
        }

    @property
    def ninvariant(self):
        """Number of invariants that are computed by the invariant() method"""
        return 1

    @tf.function
    def invariant(self, inputs):
        """Compute the quantity that is invariant under rotations

        I(t) = sum_{j=0}^{d-1} dL/d dot(q)_j

        :arg inputs: 2d-dimensional phase space vector (q,qdot)
        """
        grad_L = tf.unstack(tf.gradients(self.call(inputs), inputs)[0], axis=1)
        return tf.reduce_sum(tf.stack(grad_L[self.dim :], axis=1), axis=1)


class DoubleWellPotentialNNLagrangian(NNLagrangian):
    """Neural network representation of Lagrangian for the double well potential

    :arg dim: dimension d = number of spins
    :arg rotation_invariant: enforce rotational invariance
    """

    def __init__(self, dim, rotation_invariant=True, **kwargs):
        super(DoubleWellPotentialNNLagrangian, self).__init__(**kwargs)
        self.dim = dim
        self.rotation_invariant = rotation_invariant
        self.dense_layers = [
            tf.keras.layers.Dense(64, activation="softplus"),
            tf.keras.layers.Dense(64, activation="softplus"),
            tf.keras.layers.Dense(1, use_bias=False),
        ]

    def call(self, inputs):
        """Evaluate the Lagrangian for a given vector (q,qdot)

        :arg inputs: 2d-dimensional phase space vector (q,qdot)
        """
        if self.rotation_invariant:
            q_qdot = tf.unstack(inputs, axis=-1)
            # Extract q and qdot
            q = tf.stack(q_qdot[: self.dim], axis=-1)
            qdot = tf.stack(q_qdot[self.dim :], axis=-1)
            # Construct invariant quantities and combine them into a tensor
            x = tf.stack(
                [
                    tf.reduce_sum(tf.multiply(*pair), axis=-1)
                    for pair in combinations_with_replacement([q, qdot], 2)
                ],
                axis=-1,
            )
        else:
            x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        """Get the model configuration"""
        return {"dim": self.dim, "rotation_invariant": self.rotation_invariant}

    @property
    def ninvariant(self):
        """Number of invariants that are computed by the invariant() method"""
        return self.dim * (self.dim - 1) // 2

    @tf.function
    def invariant(self, inputs):
        """compute the dim*(dim-1)/2 angular momenta that are invariant under rotations"""
        if len(inputs.shape) < 2:
            inputs = tf.reshape(inputs, shape=[1, 2 * self.dim])
        angular_momentum = []
        q_qdot = tf.unstack(inputs, axis=-1)
        grad_L = tf.unstack(tf.gradients(self.call(inputs), inputs)[0], axis=-1)
        for j in range(self.dim):
            for k in range(j + 1, self.dim):
                angular_momentum.append(
                    tf.multiply(grad_L[self.dim + j], q_qdot[k])
                    - tf.multiply(grad_L[self.dim + k], q_qdot[j])
                )
        return angular_momentum


class TwoParticleNNLagrangian(NNLagrangian):
    """Neural network representation of Lagrangian for the two particle system

    :arg dim_space: dimension of the space
    :arg rotation_invariant: enforce rotational invariance
    :arg translation_invariant: enforce translational invariance?
    """

    def __init__(
        self, dim_space, rotation_invariant=True, translation_invariant=True, **kwargs
    ):
        super(TwoParticleNNLagrangian, self).__init__(**kwargs)
        self.dim_space = dim_space
        self.dim = 2 * dim_space
        self.rotation_invariant = rotation_invariant
        self.translation_invariant = translation_invariant
        self.dense_layers = [
            tf.keras.layers.Dense(64, activation="softplus"),
            tf.keras.layers.Dense(64, activation="softplus"),
            tf.keras.layers.Dense(1, use_bias=False),
        ]

    def call(self, inputs):
        """Evaluate the Lagrangian for a given vector (q,qdot)

        :arg inputs: 2d-dimensional phase space vector (q,qdot)
        """
        if self.rotation_invariant:
            q_qdot = tf.unstack(inputs, axis=-1)
            # Extract q and qdot
            x1 = tf.stack(q_qdot[0 : self.dim // 2], axis=-1)
            x2 = tf.stack(q_qdot[self.dim // 2 : self.dim], axis=-1)
            u1 = tf.stack(q_qdot[self.dim : 3 * self.dim // 2], axis=-1)
            u2 = tf.stack(q_qdot[3 * self.dim // 2 : 2 * self.dim], axis=-1)
            if self.translation_invariant:
                # In the translation-invariant case only x1-x2 is allowed
                dynamic_variables = [x1 - x2, u1, u2]
            else:
                dynamic_variables = [x1, x2, u1, u2]
            # Construct invariant quantities and combine them into a tensor
            x = tf.stack(
                [
                    tf.reduce_sum(tf.multiply(*pair), axis=-1)
                    for pair in list(
                        combinations_with_replacement(dynamic_variables, 2)
                    )
                ],
                axis=-1,
            )
        else:
            if self.translation_invariant:
                # Construct dx = x1 - x2
                q_qdot = tf.unstack(inputs, axis=-1)
                dx = [
                    tf.math.subtract(q_qdot[j], q_qdot[self.dim_space + j])
                    for j in range(self.dim_space)
                ]
                u = q_qdot[self.dim : 2 * self.dim]
                x = tf.stack(dx + u, axis=-1)
            else:
                x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        """Get the model configuration"""
        return {
            "dim_space": self.dim_space,
            "rotation_invariant": self.rotation_invariant,
            "translation_invariant": self.translation_invariant,
        }

    @property
    def ninvariant(self):
        """Number of invariants that are computed by the invariant() method"""
        return self.dim_space * (self.dim_space + 1) // 2

    @tf.function
    def invariant(self, inputs):
        """Compute the quantities that are invariant under *all* symmetry
        transformations of the model

        Note that depending on the values of rotation_invariant and
        translation_invariant, not all quantities might actually be conserved.

        Returns a list of conserved quantities, with the first d entries containing
        the components of the linear momentum

          M_j = dL/du^{(1)}_j + dL/du^{(2)}_j and

        the remaining d*(d-1)/2 entries containing the values of the angular momentum

          T_{j,k} = dL/du^{(1)}_j*x^{(1)}_k - dL/du^{(1)}_k*x^{(1)}_j
                  + dL/du^{(2)}_j*x^{(2)}_k - dL/du^{(2)}_k*x^{(2)}_j
        """
        if len(inputs.shape) < 2:
            inputs = tf.reshape(inputs, shape=[1, 2 * self.dim])

        q_qdot = tf.unstack(inputs, axis=-1)
        grad_L = tf.unstack(tf.gradients(self.call(inputs), inputs)[0], axis=-1)
        # Extract positions and dL/du
        x1 = q_qdot[0 : self.dim_space]
        x2 = q_qdot[self.dim_space : 2 * self.dim_space]
        dL_du1 = grad_L[self.dim : self.dim + self.dim_space]
        dL_du2 = grad_L[self.dim + self.dim_space : self.dim + 2 * self.dim_space]
        # Linar momentum
        linear_momentum = [dL_du1[j] + dL_du2[j] for j in range(self.dim_space)]
        # Angular momentum
        angular_momentum = []
        for j in range(self.dim_space):
            for k in range(j + 1, self.dim_space):
                angular_momentum.append(
                    tf.multiply(dL_du1[j], x1[k])
                    - tf.multiply(dL_du1[k], x1[j])
                    + tf.multiply(dL_du2[j], x2[k])
                    - tf.multiply(dL_du2[k], x2[j])
                )
        return linear_momentum + angular_momentum


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
        else:
            return self.dynamical_system(inputs)
