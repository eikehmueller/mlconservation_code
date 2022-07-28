from abc import abstractmethod
import tensorflow as tf
import numpy as np

"""Classes for representing Lagrangians"""


class Lagrangian(object):
    """Base class for Lagrangian

    :arg dim: dimension of state space
    """

    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def __call__(self, inputs):
        """Evaluate the Lagrangian for a given phase space vector

        :arg inputs: phase space vector (q,qdot)
        """


class HarmonicOscillatorLagrangian(Lagrangian):
    """Implements the Harmonic Oscillator Lagrangian

    L = 1/2*dot(q)^T M dot(q) - 1/2*q^T A q

    for some symmetric positive definite d x d matrices M and A.
    The state vector q is d-dimensional.

    :arg dim: dimension of state space
    :arg M_mat: mass matrix M
    :arg A_mat: Potential matrix A
    """

    def __init__(self, dim, M_mat, A_mat):
        super().__init__(dim)
        self._check_positive_definite(M_mat)
        self._check_positive_definite(A_mat)
        self.M_mat = tf.constant(M_mat, dtype=tf.float32)
        self.A_mat = tf.constant(A_mat, dtype=tf.float32)

    def _check_positive_definite(self, mat):
        """Assert that a matrix is symmetric positive definite.

        :arg mat: matrix to check
        """
        assert np.linalg.norm(mat - mat.T) < 1.0e-12
        eigenvalues, _ = np.linalg.eig(mat)
        assert all(np.real(eigenvalues) > 1.0e-6)

    @tf.function
    def __call__(self, inputs):
        """Evaluate Lagrangian

        :arg inputs: Values of q and qdot in a single tensor of shape (None,2*d)
        """
        q_qdot = tf.unstack(inputs, axis=1)
        q = tf.stack(q_qdot[: self.dim], axis=1)
        qdot = tf.stack(q_qdot[self.dim :], axis=1)
        T_kin = 0.5 * tf.einsum("aj,jk,ak->a", qdot, self.M_mat, qdot)
        V_pot = 0.5 * tf.einsum("aj,jk,ak->a", q, self.A_mat, q)
        return T_kin - V_pot


class XYModelLagrangian(Lagrangian):
    """Implements the Lagrangian of the time-dependent XY model given by

    L = a/2*sum_{j=0}^{d-1} (dq_j/dt)^2 - 1/a*sum_{j=0}^{d-1} (1-cos(q_j-q_{j-1}))

    where periodic boundary conditions q_d = q_0 are assumed.
    The state vector q is d-dimensional and we set a = 1/d.

    :arg dim: dimension of state space
    """

    def __init__(self, dim):
        super().__init__(dim)
        self.a_lat = 1.0 / self.dim

    @tf.function
    def __call__(self, inputs):
        """Evaluate Lagrangian

        :arg inputs: Values of q and qdot in a single tensor of shape (None,2*d)
        """
        q_qdot = tf.unstack(inputs, axis=1)
        q = tf.stack(q_qdot[: self.dim], axis=1)
        qdot = tf.stack(q_qdot[self.dim :], axis=1)
        T_kin = 0.5 * self.a_lat * tf.reduce_sum(tf.multiply(qdot, qdot), axis=1)
        V_pot = (
            1.0
            / self.a_lat
            * tf.reduce_sum(-tf.cos(q - tf.roll(q, shift=-1, axis=1)) + 1.0, axis=1)
        )
        return T_kin - V_pot


class DoublePendulumLagrangian(Lagrangian):
    """Implements the Lagrangian of the double pendulum given by

      L = 1/2*(m_0+m_1)*L_0^2*dot(theta_0)^2
        + 1/2*m_1*L_1^2*dot(theta_1)^2
        + m_1*L_0*L_1*dot(theta_0)*dot(theta_1)*cos(theta_0-theta_1)
        - (m_0+m_1)*g*L_0*(1-cos(theta_0))
        - m_1*g*L_1*(1-cos(theta_1))

    Here theta_0 and theta_1 are the angles relative to the vertical,
    m_0, m_1 are the masses and L_0, L_1 are the lengths of the rods.

    :arg m0: Mass of first pendulum
    :arg m1: Mass of second pendulum
    :arg L0: length of first rod
    :arg L1: length of second rod
    :arg g_grav: gravitation acceleration
    """

    def __init__(self, m0=1.0, m1=1.0, L0=1.0, L1=1.0, g_grav=9.81):
        super().__init__(2)
        self.m0 = m0
        self.m1 = m1
        self.L0 = L0
        self.L1 = L1
        self.g_grav = g_grav

    @tf.function
    def __call__(self, inputs):
        """Evaluate Lagrangian

        :arg inputs: Values of q and qdot in a single tensor of shape (None,4)
        """
        theta0, theta1, dtheta0, dtheta1 = tf.unstack(inputs, axis=1)
        T_kin = (
            1 / 2 * (self.m0 + self.m1) * self.L0**2 * dtheta0**2
            + 1 / 2 * self.m1 * self.L1**2 * dtheta1**2
            + self.m1 * self.L0 * self.L1 * dtheta0 * dtheta1 * tf.cos(theta0 - theta1)
        )
        V_pot = (self.m0 + self.m1) * self.g_grav * self.L0 * (
            1 - tf.cos(theta0)
        ) + self.m1 * self.g_grav * self.L1 * (1 - tf.cos(theta1))
        return T_kin - V_pot


class RelativisticChargedParticleLagrangian(Lagrangian):
    """Implements the Lagrangian of a relativistic particle in an electromagnetic potential

      L = 1/2*m*u^mu*u_nu + q*u^mu*A_mu(x)
        = g_{mu,nu} * (1/2*m u^mu u^nu + q u^mu A^nu)

    :arg mass: particle mass m
    :arg charge: particle charge q
    """

    def __init__(self, mass=1.0, charge=1.0):
        super().__init__(4)
        self.mass = mass
        self.charge = charge

    def __call__(self, inputs):
        # Extract velocity and vector potential
        x_u_A = tf.unstack(inputs, axis=1)
        u = tf.stack(x_u_A[4:8], axis=1)
        A_vec = tf.stack(x_u_A[8:12], axis=1)
        # Constract covariant velocity vector
        g_metric = np.diag(np.asarray([+1, -1, -1, -1], dtype=np.float32))
        u_cov = tf.tensordot(u, g_metric, axes=[[1], [0]])
        return (
            0.5 * self.mass * tf.reduce_sum(tf.multiply(u, u_cov), axis=1)
        ) + self.charge * tf.reduce_sum(tf.multiply(A_vec, u_cov), axis=1)
