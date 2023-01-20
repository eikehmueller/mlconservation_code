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
        self.dim = int(dim)

    @abstractmethod
    def call(self, y):
        """Return the acceleration d^2q/dt^2(q,qdot)

        :arg y: phase space state vector at which to evaluate the force term
        """


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
                self.acceleration_code += f"+ ({Minv_A_jk:20.14e})*q[{k:d}]"
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
        self.acceleration_code = f"""
        for (int j=0;j<{self.dim:d};++j) {{
          acceleration[j] = -{a_lat_inv2:f} * (
              sin(q[j]-q[(j+{self.dim:d}-1)%{self.dim:d}]) 
            + sin(q[j]-q[(j+1)%{self.dim:d}])
            );
        }}"""
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

    The magnetic field is always constant, but the electric field can be either constant
    (and take on the value E_electric) or vary linearly as

      E(x) = -E_electric*dot(x,E_electric/|E_electric|)

    :arg mass: mass m of particle
    :arg charge: charge q of paricle
    :arg E_electric: three vector of electric field
    :arg B_magnetic: three vector of magnetic field
    """

    def __init__(
        self,
        mass=1.0,
        charge=1.0,
        E_electric=(0.7, -1.2, 0.3),
        B_magnetic=(1.1, 0.7, 2.3),
        constant_E_electric=True,
    ):
        super().__init__(4)
        self.mass = mass
        self.charge = charge
        self.E_electric = np.asarray(E_electric)
        self.B_magnetic = np.asarray(B_magnetic)
        self.constant_E_electric = constant_E_electric
        Ex, Ey, Ez = E_electric
        Bx, By, Bz = B_magnetic
        if constant_E_electric:
            self.acceleration_code = f"""
            acceleration[0] = ({self.charge})/({self.mass}) 
                            * (({Ex})*qdot[1] + ({Ey})*qdot[2] + ({Ez})*qdot[3]);
            acceleration[1] = ({self.charge})/({self.mass})
                            * (qdot[0]*({Ex}) + qdot[2]*({Bz})-qdot[3]*({By}));
            acceleration[2] = ({self.charge})/({self.mass})
                            * (qdot[0]*({Ey}) + qdot[3]*({Bx})-qdot[1]*({Bz}));
            acceleration[3] = ({self.charge})/({self.mass})
                            * (qdot[0]*({Ez}) + qdot[1]*({By})-qdot[2]*({Bx}));
            """
        else:
            self.E_nrm_inv = 1.0 / np.linalg.norm(self.E_electric)
            self.preamble_code = "double x_E_hat;"
            self.acceleration_code = f"""
            x_E_hat = (q[1]*({Ex})+q[2]*({Ey})+q[3]*({Ez}))*({self.E_nrm_inv});
            acceleration[0] = -x_E_hat*({self.charge})/({self.mass}) 
                            * (({Ex})*qdot[1] + ({Ey})*qdot[2] + ({Ez})*qdot[3]);
            acceleration[1] = ({self.charge})/({self.mass})
                            * (-qdot[0]*({Ex})*x_E_hat + qdot[2]*({Bz})-qdot[3]*({By}));
            acceleration[2] = ({self.charge})/({self.mass})
                            * (-qdot[0]*({Ey})*x_E_hat + qdot[3]*({Bx})-qdot[1]*({Bz}));
            acceleration[3] = ({self.charge})/({self.mass})
                            * (-qdot[0]*({Ez})*x_E_hat + qdot[1]*({By})-qdot[2]*({Bx}));
            """

    def A_vec_func(self, q):
        """Vector potential of electromagnetic field

        Returns the contravariant vector A given by

          A = (-dot(q,E),1/2*cross(q,B)) for the constant electric field

        of

          A = (1/(2*|E|)*dot(q,E)^2,1/2*cross(q,B)) for the constant electric field

        :arg E_electric: electric field
        :arg B_magnetic: magnetic field
        """
        # Extract position vector
        x, y, z = tf.unstack(q, axis=-1)[1:]
        if self.constant_E_electric:
            A_0 = -(
                x * self.E_electric[0] + y * self.E_electric[1] + z * self.E_electric[2]
            )
        else:
            A_0 = (
                0.5
                * (
                    x * self.E_electric[0]
                    + y * self.E_electric[1]
                    + z * self.E_electric[2]
                )
                ** 2
                / np.linalg.norm(self.E_electric)
            )
        A_x = 0.5 * (z * self.B_magnetic[1] - y * self.B_magnetic[2])
        A_y = 0.5 * (x * self.B_magnetic[2] - z * self.B_magnetic[0])
        A_z = 0.5 * (y * self.B_magnetic[0] - x * self.B_magnetic[1])
        return tf.stack([A_0, A_x, A_y, A_z], axis=-1)

    def call(self, y):
        """Return the acceleration

        :arg y: position and velocity vector y = (x^0,x^1,x^2,x^3,u^0,u^1,u^2,u^3)
        """
        u0 = y[4]
        velocity = y[5:8]
        position = y[1:4]
        if self.constant_E_electric:
            return (
                self.charge
                / self.mass
                * np.asarray(
                    [np.dot(velocity, self.E_electric)]
                    + (
                        u0 * self.E_electric + np.cross(velocity, self.B_magnetic)
                    ).tolist()
                )
            )
        E = -self.E_electric * np.dot(position, self.E_electric) * self.E_nrm_inv
        return (
            self.charge
            / self.mass
            * np.asarray(
                [np.dot(velocity, E)]
                + (u0 * E + np.cross(velocity, self.B_magnetic)).tolist()
            )
        )


class DoubleWellPotentialSystem(DynamicalSystem):
    """Rotationally invariant double well potential in d dimensions

    The Lagrangian of a particle with mass m is given by

      L(x,u) = m/2*|u|^2 - V(x)

    where the potential has the form

      V(x) = -mu/2*|x|^2 + kappa/4*|x|^4.

    Both position x and velocity u are d-dimensional vector.
    This results in the acceleration

      du_j/dt = mu/m*x_j - kappa/m*|x|^2*x_j

    :arg dim: dimension
    :arg mass: particle mass m
    :arg mu: coefficient of the quadratic term, should be positive
    :arg kappa: coefficient of the quartic term, should be positive
    """

    def __init__(self, dim, mass=1.0, mu=1.0, kappa=1.0):
        super().__init__(dim)
        self.mass = float(mass)
        self.mu = float(mu)
        self.kappa = float(kappa)
        assert self.mu > 0
        assert self.kappa > 0
        self.preamble_code = "double q_sq;"
        self.acceleration_code = f"""
        q_sq = 0;
        for (int j=0;j<{self.dim};++j)
            q_sq += q[j]*q[j];
        for (int j=0;j<{self.dim};++j)
            acceleration[j] = (({self.mu}) - ({self.kappa})*q_sq)/({self.mass})*q[j];
        """

    def call(self, y):
        """Return the acceleration du_j/dt = mu/m*x_j - kappa/m*|x|^2*x_j

        :arg y: position and velocity vector y = (x^0,...,x^{d-1},u^0,...,u^{d-1})
        """
        q_sq = np.sum(y[: self.dim] ** 2)
        return (self.mu - self.kappa * q_sq) / self.mass * y[: self.dim]


class TwoParticleSystem(DynamicalSystem):
    """Two d-dimensional particles interacting via a quartic potential

    If x1, x2 are the particle positions and u1, u2 are their velocities, then their
    Lagrangian is given by

      L(x,u) = m1/2*|u1|^2 + m2/2*|u2|^2- V(x1,x2)

    where the potential has the form

      V(x) = -mu/2*|x|^2 + kappa/4*|x|^4.

    Note that the Lagrangian is invariant under rotations and translations
    This results in the accelerations

      du1_j/dt = (mu - kappa*|x1-x2|^2)*(x1_j-x2_j)
      du2_j/dt = (mu - kappa*|x1-x2|^2)*(x2_j-x1_j) = -du1_j/dt

    :arg dim_space: dimension of space (note that this is *half* the phase space dimension)
    :arg mass1: Mass m1 of first particle
    :arg mass2: Mass m2 of second particle
    :arg mu: coefficient of the quadratic term, should be positive
    :arg kappa: coefficient of the quartic term, should be positive
    """

    def __init__(self, dim_space, mass1=1.0, mass2=1.0, mu=1.0, kappa=1.0):
        super().__init__(2 * dim_space)
        self.mass1 = float(mass1)
        self.mass2 = float(mass2)
        self.mu = float(mu)
        self.kappa = float(kappa)
        assert self.mu > 0
        assert self.kappa > 0
        self.preamble_code = "double dq_sq;"
        self.acceleration_code = f"""
        dq_sq = 0;
        for (int j=0;j<{self.dim}/2;++j)
            dq_sq += (q[j]-q[{self.dim}/2+j])*(q[j]-q[{self.dim}/2+j]);
        for (int j=0;j<{self.dim}/2;++j) {{
            double force = (({self.mu}) - ({self.kappa})*dq_sq)*(q[j]-q[{self.dim}/2+j]);
            acceleration[j] = force/({self.mass1});
            acceleration[{self.dim}/2+j] = -force/({self.mass2});
        }}
        """

    def call(self, y):
        """Return the acceleration

        :arg y: position and velocity vector
                y = (x1^0,...,x1^{d-1},x2^0,...,x2^{d-1},u1^0,...,u1^{d-1},u2^0,...,u2^{d-1})
        """
        dq_sq = np.sum((y[0 : self.dim // 2] - y[self.dim // 2 : self.dim]) ** 2)
        force = (self.mu - self.kappa * dq_sq) * (
            y[0 : self.dim // 2] - y[self.dim // 2 : self.dim]
        )
        return np.concatenate([force / self.mass1, -force / self.mass2])


class KeplerSystem(DynamicalSystem):
    """Motion of a non-relativistic particle under a 1/r central field

    The acceleration is given by

    du_j/dt = -alpha/m * x_j/|x|^3

    Both position x and velocity u are 3-dimensional vectors.
    This results in the acceleration

    :arg mass: particle mass m
    :arg alpha: coefficient of the 1/r term
    """

    def __init__(self, mass=1.0, alpha=1.0):
        super().__init__(3)
        self.mass = float(mass)
        self.alpha = float(alpha)
        assert self.mass > 0
        assert self.alpha > 0
        self.header_code = "#include <math.h>"
        self.preamble_code = "double q_sq, inv_q_three_half;"
        self.acceleration_code = f"""
        q_sq = 0;
        for (int j=0;j<{self.dim};++j)
            q_sq += q[j]*q[j];
        inv_q_three_half = 1./(q_sq*sqrt(q_sq));
        for (int j=0;j<{self.dim};++j)
            acceleration[j] = -({self.alpha})/({self.mass})*inv_q_three_half*q[j];
        """

    def call(self, y):
        """Return the acceleration du_j/dt = -alpha/m*x_j/|x|^3

        :arg y: position and velocity vector y = (x^0,x^1,x^2,u^0,u^1,u^2)
        """
        q_sq = np.sum(y[:3] ** 2)
        inv_q_three_half = 1.0 / (q_sq * np.sqrt(q_sq))
        return -self.alpha / self.mass * inv_q_three_half * y[:3]


class SchwarzschildSystem(DynamicalSystem):
    """Motion of a relativistic particle in the Schwarzschild metric

    Let q = (x^0,x) and dot(q) = (dx^0/dt,u), where x, u are the contra-variant position and
    velocity and write r = |x|. Then

      d^2x^j/dt^2 = -r_s/(2*r^3)*(1+3*(r^2*u^2-(x.u)^2)/r^2) * x^j  for j=1,2,3
      d^2x^0/dt^2 = r_s/(2*r)*(1-r_s/r)^{-1} (x.u)/r^2 dx^0/dt

    :arg r_s: Schwarzschild radius
    """

    def __init__(self, r_s=1.0):
        super().__init__(4)
        self.r_s = float(r_s)
        assert self.r_s > 0
        self.header_code = "#include <math.h>"
        self.preamble_code = "double r_sq, u_sq, x_dot_u, rs_over_r, r_inv_sq, tmp_x;"
        self.acceleration_code = f"""
        r_sq = 0;
        u_sq = 0;
        x_dot_u = 0;
        for (int j=1;j<{self.dim};++j) {{
            r_sq += q[j]*q[j];
            u_sq += qdot[j]*qdot[j];
            x_dot_u += q[j]*qdot[j];
        }}
        r_inv_sq = 1.0/r_sq;
        rs_over_r = ({self.r_s})*sqrt(r_inv_sq);
        tmp_x = -0.5*rs_over_r*r_inv_sq*(1+3*(r_sq*u_sq-x_dot_u*x_dot_u)*r_inv_sq);
        for (int j=1;j<{self.dim};++j)
            acceleration[j] = tmp_x*q[j];
        acceleration[0] = -rs_over_r/(1-rs_over_r)*x_dot_u*r_inv_sq*qdot[0];
        """

    def call(self, y):
        """Return the acceleration

        :arg y: position and velocity vector y = (x^0,x^1,x^2,x^3,u^0,u^1,u^2,u^3)
        """
        r_sq = np.sum(y[1:4] ** 2)
        u_sq = np.sum(y[5:8] ** 2)
        x_dot_u = np.sum(y[1:4] * y[5:8])
        r_inv_sq = 1 / r_sq
        rs_over_r = self.r_s * np.sqrt(r_inv_sq)
        acceleration = np.zeros(shape=[4])
        acceleration[1:4] = (
            -0.5
            * rs_over_r
            * r_inv_sq
            * (1 + 3 * (r_sq * u_sq - x_dot_u**2) * r_inv_sq)
            * y[1:4]
        )
        acceleration[0] = -rs_over_r / (1 - rs_over_r) * x_dot_u * r_inv_sq * y[4]
        return acceleration
