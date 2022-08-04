"""User defined neural network layers"""
import math
import numpy as np
import tensorflow as tf
from itertools import permutations, combinations, combinations_with_replacement
from sympy.combinatorics.permutations import Permutation


class RotationallyInvariantLayer(tf.keras.layers.Layer):
    """Layer for reducing n tensors a_0,a_1,...,a_{n-1} into rotationally invariant combinations

    Each tensor a_j is assumed to represent a vector of dimension d
    (possibly replicated along the batch dimension).

    The rotationally invariant combinations are given by

      1. the dot products of all possible pairings of the tensors
      2. contractions of all selections of d tensors with the d-dimensional Levi-Civita symbol

    There are (n choose 2) + n = n*(n+1)/2 dot products and (n choose d) Levi-Civita contractions
    (if n=d there is exactly one contraction and there are no contractions if n<d).

    If we also assume invariance under reflections (i.e. the full O(d) group instead of SO(d)), then
    only dot products are included.

    :arg dim_space: dimension of vectors
    :arg n_tensors: number of tensors in input
    :arg reflection_invariant: do we assume invariance under reflections?
    """

    def __init__(self, dim_space, n_tensors, reflection_invariant=False, **kwargs):
        super(RotationallyInvariantLayer, self).__init__(**kwargs)
        self.dim_space = dim_space
        self.n_tensors = n_tensors
        self.reflection_invariant = reflection_invariant

    @property
    def n_outputs(self):
        "number of output tensors" ""
        n_dot_products = math.comb(self.n_tensors, 2) + self.n_tensors
        n_cross_products = math.comb(self.n_tensors, self.dim_space)
        if self.reflection_invariant:
            return n_dot_products
        else:
            return n_dot_products + n_cross_products

    def call(self, inputs):
        """Combine inputs into rotationally invariant combinations

        The input is assumed to be of shape (BATCHSIZE,d*n) where d
        is the dimension of the vector space and n is the number of
        tensors that are to be contracted. Hence, it represents n tensors
        of shape (BATCHSIZE,d) or a_0,a_1,...a_{n-1}

        :arg inputs: a tensor of shape
        """
        # ==== 1. dot products a_j.a_k ====
        z = tf.unstack(inputs, axis=-1)
        # Ensure that input is of the correct size
        assert len(z) == self.dim_space * self.n_tensors
        # a_vec[j] is a tensor of shape (BATCHSIZE,d) for j = 0,1,...n-1
        a_vec = [
            tf.stack(z[j * self.dim_space : (j + 1) * self.dim_space], axis=-1)
            for j in range(self.n_tensors)
        ]
        invariants = [
            tf.reduce_sum(tf.multiply(*pair), axis=-1)
            for pair in list(combinations_with_replacement(a_vec, 2))
        ]
        if not self.reflection_invariant:
            # === 2. Contractions with Levi-Civita tensor ===
            # b_vec_list[j][k] is the k-th entry of the j-the tensor where j=0,1,...n-1 and k=0,1,...,d-1
            b_vec_list = [
                z[j * self.dim_space : (j + 1) * self.dim_space]
                for j in range(self.n_tensors)
            ]
            # Loop over all combinations of d vectors
            for b_vec in combinations(b_vec_list, self.dim_space):
                # Construct the product t = sum_{permutations p} S(p) * prod_j b^{(j)}_{p(j)}
                # where S(p) is the signature of the permutation
                pt = []
                for p in permutations(range(self.dim_space)):
                    t = tf.math.scalar_mul(
                        Permutation.signature(Permutation(p)), b_vec[0][p[0]]
                    )
                    for j in range(1, self.dim_space):
                        t = tf.math.multiply(t, b_vec[j][p[j]])
                    pt.append(t)
                invariants.append(tf.math.add_n(pt))
        return tf.stack(invariants, axis=-1)
