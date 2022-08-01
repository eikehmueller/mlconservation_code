import numpy as np


class XYModelRandomInitializer(object):
    """Random initialiser class for the XY model"""

    def __init__(self, dim):
        self.dim = dim

    def draw(self):
        """Draw a new sample with

        q_j ~ Uniform(-pi,+pi)
        qdot_j ~ Normal(0,1)
        """
        q = np.random.uniform(low=-np.pi, high=+np.pi, size=self.dim)
        qdot = np.random.normal(size=self.dim)
        return q, qdot


class XYModelConstantInitializer(object):
    """Constant initialiser class for the XY model"""

    def __init__(self, dim):
        self.dim = dim

    def draw(self):
        """Draw a new sample with

        q_j = 0
        qdot_j = j/d
        """

        q = np.zeros(self.dim)
        qdot = np.arange(0, self.dim) / self.dim
        return q, qdot


class RelativisticChargedParticleRandomInitializer(object):
    """Random initializer for charged particle system

    :arg rho: scaling-factor for three-velocity
    """

    def __init__(self, rho):
        self.dim = 4
        self.rho = rho

    def draw(self):
        """Draw a new sample with

          q_j = 0
          qdot_0 = 1
          qdot_j = rho*v_j/|v|   with  rho

        where v is a normal random vector. This choice guarantees that qdot^2 = 1 - rho^2 > 0
        """
        q = np.zeros(shape=[self.dim])
        qdot = np.zeros(shape=[self.dim])
        velocity = np.random.normal(size=3)
        velocity = self.rho * velocity / np.linalg.norm(velocity)
        qdot[0] = 1
        qdot[1:4] = velocity
        return q, qdot


class DoubleWellPotentialConstantInitializer(object):
    """Constant initialiser class for the double well potential model"""

    def __init__(self, dim):
        self.dim = dim
        assert dim <= 8, "only dimensions up to 8 are supported"
        self.q_ref = np.asarray(
            [
                0.73906985,
                -0.29971694,
                0.31880467,
                0.42600132,
                -0.16290228,
                -0.19736723,
                0.21630599,
                1.13949553,
            ]
        )
        self.qdot_ref = np.asarray(
            [
                0.11062122,
                -0.61520255,
                1.32813101,
                0.28267341,
                0.8595746,
                0.18834262,
                0.69556394,
                -0.13685782,
            ]
        )

    def draw(self):
        """Draw a new sample with

        q_j = 0
        qdot_j = j/d
        """
        return self.q_ref[: self.dim], self.qdot_ref[: self.dim]


class TwoParticleConstantInitializer(object):
    """Constant initialiser class for interacting two particle system"""

    def __init__(self, dim):
        self.dim = dim
        assert dim <= 8, "only dimensions up to 8 are supported"
        self.q_ref = np.asarray(
            [
                0.73906985,
                -0.29971694,
                0.31880467,
                0.42600132,
                -0.16290228,
                -0.19736723,
                0.21630599,
                1.13949553,
            ]
        )
        self.qdot_ref = np.asarray(
            [
                0.11062122,
                -0.61520255,
                1.32813101,
                0.28267341,
                0.8595746,
                0.18834262,
                0.69556394,
                -0.13685782,
            ]
        )

    def draw(self):
        """Draw a new sample with

        q_j = 0
        qdot_j = j/d
        """
        return self.q_ref[: self.dim], 0.05 * self.qdot_ref[: self.dim]
