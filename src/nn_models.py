import os
import errno
import json
import numpy as np
import tensorflow as tf
from itertools import combinations_with_replacement
from auxilliary import ndarrayDecoder, ndarrayEncoder
from lagrangian_dynamical_system import LagrangianDynamicalSystem
from nn_layers import RotationallyInvariantLayer


class NNLagrangian(tf.keras.layers.Layer):
    """Base class for neural network based Lagrangians"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        config["dense_layers"] = None
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
    :arg dense_layers: dense layers
    :arg rotation_invariant: enforce rotational invariance
    :arg shift_invariant: enforce shift invariance
    """

    def __init__(
        self, dim, dense_layers, rotation_invariant=True, shift_invariant=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.rotation_invariant = rotation_invariant
        self.shift_invariant = shift_invariant
        self.dense_layers = [] if dense_layers is None else dense_layers
        self.dense_layers.append(tf.keras.layers.Dense(1, use_bias=False))

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
            qdot = tf.roll(tf.stack(q_qdot[self.dim :], axis=1), offset, 1)
            x = tf.concat([q, qdot], 1)
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


class SingleParticleNNLagrangian(NNLagrangian):
    """Neural network representation of Lagrangian for a single particle moving in d dimensions

    If rotation_invariant is True, invariance under rotations (i.e. the SO(d) group) is assumed.
    If in addition reflection_invariant is True, then we also assume invariance under reflections,
    i.e. the larger group O(d)

    :arg dim: space dimension d
    :arg dense_layers: intermediate dense layers
    :arg rotation_invariant: enforce rotational invariance
    :arg reflection_invariant: enforce invariance under reflections
    """

    def __init__(
        self,
        dim,
        dense_layers,
        rotation_invariant=True,
        reflection_invariant=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.rotation_invariant = rotation_invariant
        self.reflection_invariant = reflection_invariant
        # Add final layer
        self.dense_layers = [] if dense_layers is None else dense_layers
        self.dense_layers.append(tf.keras.layers.Dense(1, use_bias=False))

    def call(self, inputs):
        """Evaluate the Lagrangian for a given vector (q,qdot)

        :arg inputs: 2d-dimensional phase space vector (q,qdot)
        """
        if self.rotation_invariant:
            x = RotationallyInvariantLayer(self.dim, 2, self.reflection_invariant)(
                inputs
            )
        else:
            x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        """Get the model configuration"""
        return {
            "dim": self.dim,
            "rotation_invariant": self.rotation_invariant,
            "reflection_invariant": self.reflection_invariant,
        }

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


class SchwarzschildNNLagrangian(NNLagrangian):
    """Neural network representation of Lagrangian for a single relativistc particle
    moving in a rotationally invariant metric

    If rotation_invariant is True, invariance under spatial rotations (i.e. the O(3) group)
    is assumed. In this case, the invariants are the temporal components of the four-dimensional
    position- and velocity vectors as well as any dot-products between the three vectors.

    :arg dense_layers: intermediate dense layers
    :arg rotation_invariant: enforce rotational invariance
    """

    def __init__(self, dense_layers, rotation_invariant=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = 4
        self.rotation_invariant = rotation_invariant
        # Add final layer
        self.dense_layers = [] if dense_layers is None else dense_layers
        self.dense_layers.append(tf.keras.layers.Dense(1, use_bias=False))

    def call(self, inputs):
        """Evaluate the Lagrangian for a given vector (q,qdot)

        :arg inputs: 2d-dimensional phase space vector (q,qdot)
        """
        if self.rotation_invariant:
            q_qdot = tf.unstack(inputs, axis=-1)
            # temporal component and its derivative
            t = q_qdot[0]
            t_dot = q_qdot[4]
            # three-dimensional position and three-velocity
            x_spat = tf.stack(q_qdot[1:4], axis=-1)
            v_spat = tf.stack(q_qdot[5:8], axis=-1)
            # construct the five invariants
            invariants = [
                tf.reduce_sum(tf.multiply(*pair), axis=-1)
                for pair in list(combinations_with_replacement([x_spat, v_spat], 2))
            ] + [t, t_dot]
            x = tf.stack(invariants, axis=-1)
        else:
            x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        """Get the model configuration"""
        return {
            "rotation_invariant": self.rotation_invariant,
        }

    @property
    def ninvariant(self):
        """Number of invariants that are computed by the invariant() method"""
        return 3

    @tf.function
    def invariant(self, inputs):
        """compute the three components of angular momentum"""
        if len(inputs.shape) < 2:
            inputs = tf.reshape(inputs, shape=[1, 8])
        angular_momentum = []
        q_qdot = tf.unstack(inputs, axis=-1)
        grad_L = tf.unstack(tf.gradients(self.call(inputs), inputs)[0], axis=-1)
        for j in range(1, 4):
            for k in range(j + 1, 4):
                angular_momentum.append(
                    tf.multiply(grad_L[4 + j], q_qdot[k])
                    - tf.multiply(grad_L[4 + k], q_qdot[j])
                )
        return angular_momentum


class TwoParticleNNLagrangian(NNLagrangian):
    """Neural network representation of Lagrangian for the two particle system

    If rotation_invariant is True, invariance under rotations (i.e. the SO(d) group) is assumed.
    If in addition reflection_invariant is True, then we also assume invariance under reflections,
    i.e. the larger group O(d)

    :arg dim_space: dimension of the space
    :arg dense_layers: intermediate dense layers
    :arg rotation_invariant: enforce rotational invariance
    :arg translation_invariant: enforce translational invariance?
    :arg reflection_invariant: enforce invariance under reflections
    """

    def __init__(
        self,
        dim_space,
        dense_layers,
        rotation_invariant=True,
        translation_invariant=True,
        reflection_invariant=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_space = dim_space
        self.dim = 2 * dim_space
        self.rotation_invariant = rotation_invariant
        self.translation_invariant = translation_invariant
        self.reflection_invariant = reflection_invariant
        self.dense_layers = [] if dense_layers is None else dense_layers
        self.dense_layers.append(tf.keras.layers.Dense(1, use_bias=False))

    def call(self, inputs):
        """Evaluate the Lagrangian for a given vector (q,qdot)

        :arg inputs: 2d-dimensional phase space vector (q,qdot)
        """
        if self.translation_invariant:
            # Construct dx = x1 - x2
            q_qdot = tf.unstack(inputs, axis=-1)
            dx = [
                tf.math.subtract(q_qdot[j], q_qdot[self.dim_space + j])
                for j in range(self.dim_space)
            ]
            u = q_qdot[self.dim : 2 * self.dim]
            x = tf.stack(dx + u, axis=-1)
            n_tensors = 3
        else:
            x = inputs
            n_tensors = 4
        if self.rotation_invariant:
            x = RotationallyInvariantLayer(
                self.dim_space, n_tensors, self.reflection_invariant
            )(x)
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        """Get the model configuration"""
        return {
            "dim_space": self.dim_space,
            "rotation_invariant": self.rotation_invariant,
            "translation_invariant": self.translation_invariant,
            "reflection_invariant": self.reflection_invariant,
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
