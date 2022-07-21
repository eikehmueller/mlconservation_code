from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class DynamicalSystem(ABC):
    def __init__(self, dim):
        """
        represents dynamical systems of the form

        dq/dt = qdot
        dqdot/dt = f(q,qdot)

        :arg dim: dimension of state space. Note that
        """
        self.dim = dim

    @abstractmethod
    def call(self, y):
        """Return the acceleration d^2q/dt^2(q,qdot)

        :arg y: phase space state vector at which to evaluate the force term
        """


class LagrangianDynamicalSystem(tf.keras.layers.Layer):
    """

    The time evolution of the variable y = (q, dot{q})
    can be written as

    dy/dt = f(y)

    This class implement the function f for a given Lagrangian L.

    In this case the function can be written as

    f = (dot{q}, g)

    with

    g_k = ( (dL/dq)_j - dot{q}_i*(J_{q,dot{q}})_{ij} )*(J_{dot{q},dot{q}}^{-1})_{jk}

    and the matrices (J_{q,dot{q}})_{ij} = d^2 L / (dq_i ddot{q}_j),
    (J_{dot{q},dot{q}})_{ij} = d^2 L / (ddot{q}_i ddot{q}_j)

    """

    def __init__(self, lagrangian):
        super().__init__()
        self.dim = lagrangian.dim
        self.lagrangian = lagrangian

    @tf.function
    def _hessian(self, y, j, k):
        """Helper function for computing Hessian d^2 L / (dy_j dy_k)"""
        d_y_j = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)[j]
        return tf.unstack(tf.gradients(d_y_j, y)[0], axis=1)[k]

    @tf.function
    def div_L(self, y):
        """Gradient of Lagrangian dL/dq"""
        dL = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)
        return tf.stack(dL[: self.dim], axis=1)

    @tf.function
    def J_qdotqdot(self, y):
        """d x d matrix J_{dot{q},dot{q}}"""
        rows = []
        for j in range(self.dim, 2 * self.dim):
            row = []
            for k in range(self.dim, 2 * self.dim):
                row.append(self._hessian(y, j, k))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def J_qqdot(self, y):
        """d x d matrix J_{q,dot{q}}"""
        rows = []
        for j in range(0, self.dim):
            row = []
            for k in range(self.dim, 2 * self.dim):
                row.append(self._hessian(y, j, k))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def J_qdotqdot_inv(self, y):
        """d x d matrix (J_{dot{q},dot{q}})^{-1}"""
        return tf.linalg.inv(self.J_qdotqdot(y))

    @tf.function
    def call(self, y):
        """Return value of the acceleration d^2q/dt^2(q,qdot) for a given input y

        :arg y: Phase space vector y = (q,qdot)
        """
        q_qdot = tf.unstack(y, axis=1)
        qdot = tf.stack(q_qdot[self.dim :], axis=1)
        qdotdot = tf.einsum(
            "ai,aij->aj",
            self.div_L(y) - tf.einsum("ai,aij->aj", qdot, self.J_qqdot(y)),
            self.J_qdotqdot_inv(y),
        )
        return qdotdot


class HarmonicOscillatorSystem(DynamicalSystem):
    """Dynamical system with force term derived from the
    Harmonic Oscillator Lagrangian

        L = 1/2*dot(q)^T M dot(q) - 1/2*q^T A q

    The rate of change of qdot is given by

        F(q) = - M^{-1} A q
    """

    def __init__(self, dim, M_mat, A_mat):
        """Construct a new instance

        :arg dim: dimension of state space
        :arg M_mat: mass matrix M
        :arg A_mat: Potential matrix A
        """
        super().__init__(dim)
        self.M_mat = M_mat
        self.A_mat = A_mat
        self.Minv_A = np.linalg.inv(M_mat) @ A_mat
        # Generate the C-code used for the acceleration update
        self.acceleration_code = ""
        for j in range(self.dim):
            self.acceleration_code += f"acceleration[{j:d}] = -("
            for k in range(self.dim):
                Minv_A_jk = self.Minv_A[j, k]
                self.acceleration_code += f"+ ({Minv_A_jk:e})*q[{k:d}]"
            self.acceleration_code += ");\n"

    def call(self, y):
        """Return the rate of change of qdot

        :arg y: phase space state vector (q,qdot) at which to evaluate the acceleration
        """
        q = y[: self.dim]
        return -self.Minv_A @ q


class XYModelSystem(DynamicalSystem):
    """Dynamical system with force term derived from the
    time dependent XY Model

        dqdot/dt_j = - 1/a^2 * (sin(q_j-q_{j-1})+sin(q_j-q_{j+1}))
    """

    def __init__(self, dim):
        """Construct a new instance

        :arg dim: dimension of state space.
        """
        super().__init__(dim)
        self.a_lat = 1.0 / self.dim
        # Generate the C-code used for the acceleration update
        a_lat_inv2 = 1.0 / self.a_lat**2
        self.acceleration_code = f"for (int j=0;j<{self.dim:d};++j)" + "{\n"
        self.acceleration_code += f"  acceleration[j] = -{a_lat_inv2:f} * ("
        self.acceleration_code += f"sin(q[j]-q[(j+{self.dim:d}-1)%{self.dim:d}])"
        self.acceleration_code += f"+sin(q[j]-q[(j+1)%{self.dim:d}])" + ");\n"
        self.acceleration_code += "}"
        self.header_code = "#include <math.h>"

    def call(self, y):
        """Return the accelerationrate of change of qdot f(y)

        :arg y: phase space state vector (q,qdot) at which to evaluate the
                acceleration
        """
        q = y[: self.dim]
        return (
            -1.0
            / self.a_lat**2
            * (np.sin(q - np.roll(q, +1)) + np.sin(q - np.roll(q, -1)))
        )


class DoublePendulumSystem(DynamicalSystem):
    """Dynamical system for double pendulum.

    The equations of motion are given by

      d^2theta_0/dt^2 = (f_0-alpha_0*f_1) / (1-alpha_0*alpha_1)
      d^2theta_1/dt^2 = (-alpha_1*f_0+f_1) / (1-alpha_0*alpha_1)

    with

    alpha_0 = L_1/L_0 * m_1/(m_0+m_1)*cos(theta_0-theta_1)
    alpha_1 = L_0/L_1 * cos(theta_0-theta_1)

    f_0 = -L_1/L_0 * m_1/(m_0+m_1) * dot(theta_1)^2*sin(theta_0-theta_1)
          - g/L_0*sin(theta_0)

    f_1 = L_0/L_1 * dot(theta_0)^2*sin(theta_0-theta_1) - g/L_1*sin(theta_1)

        Here theta_0 and theta_1 are the angles relative to the vertical,
    m_0, m_1 are the masses and L_0, L_1 are the lengths of the rods.

    For a derivation of the equations of motion see for example

    https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6

    :arg m0: Mass of first pendulum
    :arg m1: Mass of second pendulum
    :arg L0: length of first rod
    :arg L1: length of second rod
    :arg g_grav: gravitation acceleration
    """

    def __init__(self, m0=1.0, m1=1.0, L0=1.0, L1=1.0, g_grav=9.81):
        """Construct a new instance

        :arg dim: dimension of state space.
        """
        super().__init__(2)
        self.m0 = m0
        self.m1 = m1
        self.L0 = L0
        self.L1 = L1
        self.g_grav = g_grav
        self.header_code = "#include <math.h>"
        self.preamble_code = """
        double alpha0, alpha1;
        double f0, f1;
        double rho;
        """
        self.acceleration_code = f"""
        alpha0 = ({self.L1})/({self.L0})
               * ({self.m1})/({self.m0}+{self.m1})
               * cos(q[0]-q[1]);
        alpha1 = ({self.L0})/({self.L1})
               * cos(q[0]-q[1]);
        f0 = -({self.L1})/({self.L0})
           * ({self.m1})/({self.m0}+{self.m1})
           * qdot[1]*qdot[1]
           * sin(q[0]-q[1])
           - ({self.g_grav})/({self.L0})*sin(q[0]);
        f1 = ({self.L0})/({self.L1})
           * qdot[0]*qdot[0]
           * sin(q[0]-q[1])
           - ({self.g_grav})/({self.L1})*sin(q[1]);
        rho = 1./(1.-alpha0*alpha1);
        acceleration[0] = rho*(f0-alpha0*f1);
        acceleration[1] = rho*(f1-alpha1*f0);
        """

    def call(self, y):
        """Return the acceleration (rate of change of qdot) a(y)

        :arg y: phase space state vector (q,qdot) at which to evaluate the
                acceleration
        """
        theta0, theta1, dtheta0, dtheta1 = y[:]
        alpha0 = (
            self.L1 / self.L0 * self.m1 / (self.m0 + self.m1) * np.cos(theta0 - theta1)
        )
        alpha1 = self.L0 / self.L1 * np.cos(theta0 - theta1)
        f0 = -self.L1 / self.L0 * self.m1 / (self.m0 + self.m1) * dtheta1**2 * np.sin(
            theta0 - theta1
        ) - self.g_grav / self.L0 * np.sin(theta0)
        f1 = self.L0 / self.L1 * dtheta0**2 * np.sin(
            theta0 - theta1
        ) - self.g_grav / self.L1 * np.sin(theta1)
        rho = 1 / (1 - alpha0 * alpha1)
        return np.array([rho * (f0 - alpha0 * f1), rho * (f1 - alpha1 * f0)])
