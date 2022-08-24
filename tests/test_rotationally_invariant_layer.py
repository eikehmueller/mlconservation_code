"""Check that the rotationally invariant layer contracts tensors correctly"""
import pytest
import numpy as np
import tensorflow as tf
from itertools import permutations, combinations, combinations_with_replacement
from sympy.combinatorics.permutations import Permutation
from conservative_nn.nn_layers import RotationallyInvariantLayer

from common import rng


@pytest.mark.parametrize("dim_space", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("n_tensors", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("reflection_invariant", [True, False])
def test_rotationally_invariant_layer(dim_space, rng, n_tensors, reflection_invariant):
    """Compare the output from a rotationally invariant layer to a manual calculation

    :arg dim_space: dimension d of space
    :arg n_tensors: number n of tensors
    :arg reflection_invariant: assume invariance under reflections?
    """
    # Tolerance for comparison
    tolerance = 1.0e-12
    # Create tensor of shape (BATCHSIZE,dim_space * n_tensors)
    BATCHSIZE = 1
    inputs = rng.standard_normal(size=dim_space * n_tensors)

    # Manually reduce
    # Split inputs into n vectors of size d
    a_vec = [inputs[j * dim_space : (j + 1) * dim_space] for j in range(n_tensors)]
    # 1. Construct dot products
    dot_products = [
        np.dot(*pair) for pair in list(combinations_with_replacement(a_vec, 2))
    ]
    # 2. Contract contractions with Levi Civita symbol
    cross_products = []
    # Loop over all combinations
    for b_vec in combinations(a_vec, dim_space):
        pt = 0
        for p in permutations(range(dim_space)):
            t = Permutation.signature(Permutation(p))
            for j in range(dim_space):
                t *= b_vec[j][p[j]]
            pt += t
        cross_products.append(pt)
    if reflection_invariant:
        result_np = np.asarray(dot_products)
    else:
        result_np = np.asarray(dot_products + cross_products)

    rot_inv_layer = RotationallyInvariantLayer(
        dim_space, n_tensors, reflection_invariant=reflection_invariant
    )
    inputs_tf = tf.constant(inputs, shape=[BATCHSIZE, dim_space * n_tensors])
    result_tf = rot_inv_layer(inputs_tf).numpy().flatten()
    assert np.linalg.norm(result_tf - result_np) < tolerance
