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
