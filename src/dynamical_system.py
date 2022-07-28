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


class RelativisticChargedParticleLagrangianDynamicalSystem(tf.keras.layers.Layer):
    """Dynamical system derived from a relativistic Lagrangian with vector potential.

    Given (contravariant) coordinates x^mu, velocities u^mu and a vector potential
    A^mu (which depends on the position x), the time evolution of the
    state y = (x,u) = (q,dot{q}) can be written as

      dy/dt = f(q,dot{q})

    for some function

      f = (dot{q}, g)

    The components of the function g are given by:

    g^mu = ( dL/dx^nu + dA^rho/dx^nu * dL/dA^rho
             - u^rho* ((J_{x,u})_{rho,nu} + dA^sigma/dx^rho * (J_{A,u})_{sigma,nu}) ) * (J^{-1}_{u,u})^{nu,mu}

    where the matrices J_{x,u}, J_{u,u} and J_{A,u} are defined as

    (J_{x,u})_{mu,nu} = d^2L/(dx^mu du^nu)
    (J_{u,u})_{mu,nu} = (d^L/(du^mu du^nu))
    (J_{A,u})_{mu,nu} = d^2L/(dA^mu du^nu)

    It is assumed that the functional dependence of the vector potential
    is known and passed as a tensorflow function.

    :arg lagrangian: Lagrangian L(x,u,A)
    :arg A_vec_func: Vector potential function A(x)
    """

    def __init__(self, lagrangian, A_vec_func):
        super().__init__()
        self.dim = 4
        self.lagrangian = lagrangian
        self.A_vec_func = A_vec_func

    @tf.function
    def _hessian(self, y, mu, nu):
        """Helper function for computing Hessian d^2 L / (dy^mu dy^nu)"""
        d_y_mu = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)[mu]
        return tf.unstack(tf.gradients(d_y_mu, y)[0], axis=1)[nu]

    @tf.function
    def dL_dx(self, y):
        """Gradient of Lagrangian dL/dx"""
        dL = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)
        return tf.stack(dL[:4], axis=1)

    @tf.function
    def dL_dA(self, y):
        """Partial derivative of Lagrangian with respect to A, dL/dA"""
        dL = tf.unstack(tf.gradients(self.lagrangian(y), y)[0], axis=1)
        return tf.stack(dL[8:], axis=1)

    @tf.function
    def J_xu(self, y):
        """4 x 4 matrix J_{x,u}"""
        rows = []
        for mu in range(0, 4):
            row = []
            for nu in range(4, 8):
                row.append(self._hessian(y, mu, nu))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def J_uu(self, y):
        """4 x 4 matrix J_{u,u}"""
        rows = []
        for mu in range(4, 8):
            row = []
            for nu in range(4, 8):
                row.append(self._hessian(y, mu, nu))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def J_Au(self, y):
        """4 x 4 matrix J_{A,u}"""
        rows = []
        for mu in range(8, 12):
            row = []
            for nu in range(4, 8):
                row.append(self._hessian(y, mu, nu))
            rows.append(tf.stack(row, axis=1))
        return tf.stack(rows, axis=1)

    @tf.function
    def dA_dx(self, x):
        """Compute the derivative of A^mu with respect to x^nu"""
        A_vec = tf.unstack(self.A_vec_func(x), axis=1)
        dA_dx_list = []
        for i in range(4):
            dA_dx_list.append(tf.gradients(A_vec[i], x, stop_gradients=[x])[0])
        return tf.stack(dA_dx_list, axis=2)

    @tf.function
    def J_uu_inv(self, y):
        """4 x 4 matrix (J_{uu})^{-1}"""
        return tf.linalg.inv(self.J_uu(y))

    @tf.function
    def call(self, y):
        """Return value of the acceleration d^2u/dt^2(x,u,A) for a given input y = (x,u,A)

        :arg y: Phase space vector y = (q,qdot) = (x,u)
        """
        # Compute x, u and A(x) which can then be fed to the Lagrangian
        x_u = tf.unstack(y, axis=1)
        x = tf.stack(x_u[0:4], axis=1)
        u = tf.stack(x_u[4:8], axis=1)
        A_vec = tf.unstack(self.A_vec_func(x), axis=1)
        x_u_A = tf.stack(x_u + A_vec, axis=1)
        acceleration = tf.einsum(
            "ai,aij->aj",
            self.dL_dx(x_u_A)
            + tf.einsum("ajk,ak->aj", self.dA_dx(x), self.dL_dA(x_u_A))
            - tf.einsum(
                "ai,aij->aj",
                u,
                self.J_xu(x_u_A)
                + tf.einsum("ajk,akm->ajm", self.dA_dx(x), self.J_Au(x_u_A)),
            ),
            self.J_uu_inv(x_u_A),
        )
        return acceleration


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


class RelativisticChargedParticleSystem(DynamicalSystem):
    """Relativistic particle moving in constant electromagnetic field.

    The relativistic equation of motion of the particle is given by:

      m du^mu/dt = q u^nu (g^{mu,rho} g_{nu,sigma} dA^sigma/dx^rho - dA^mu/dx^nu)

    where g = diag(+1,-1,-1,-1) is the metric tensor.
    For a constant magnetic field pointing this reduces to:

      du^0 = q E.v
      du^j/dt = q/m*(u^0 E + v x B)^j

    where v = (u^1,u^2,u^3) are the spatial components of the velocity and hat(B) is
    the direction of the magnetic field.

    :arg mass: mass m of particle
    :arg charge: charge q of paricle
    :arg E_electric: three vector of electric field
    :arg B_magnetic: three vector of magnetic field
    """

    def __init__(
        self,
        mass=1.0,
        charge=1.0,
        E_electric=[0.7, -1.2, 0.3],
        B_magnetic=[1.1, 0.7, 2.3],
    ):
        super().__init__(4)
        self.mass = mass
        self.charge = charge
        self.E_electric = np.asarray(E_electric)
        self.B_magnetic = np.asarray(B_magnetic)
        Ex, Ey, Ez = E_electric
        Bx, By, Bz = B_magnetic
        self.acceleration_code = f"""
        acceleration[0] = ({self.charge})/({self.mass})*(({Ex})*qdot[1] + ({Ey})*qdot[2] + ({Ez})*qdot[3]);
        acceleration[1] = ({self.charge})/({self.mass})*(qdot[0]*({Ex}) + qdot[2]*({Bz})-qdot[3]*({By}));
        acceleration[2] = ({self.charge})/({self.mass})*(qdot[0]*({Ey}) + qdot[3]*({Bx})-qdot[1]*({Bz}));
        acceleration[3] = ({self.charge})/({self.mass})*(qdot[0]*({Ez}) + qdot[1]*({By})-qdot[2]*({Bx}));
        """

    def call(self, y):
        """Return the acceleration

        :arg y: position and velocity vector y = (x^0,x^1,x^2,x^3,u^0,u^1,u^2,u^3)
        """
        u0 = y[4]
        velocity = y[5:8]
        return (
            self.charge
            / self.mass
            * np.asarray(
                [np.dot(velocity, self.E_electric)]
                + (u0 * self.E_electric + np.cross(velocity, self.B_magnetic)).tolist()
            )
        )
