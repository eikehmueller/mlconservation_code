import json
import numpy as np


class NormalRandomLookup:
    """Lookup table for normally distributed random numbers that have been
    saved to disk

    Class for reading json files which contain a single list of numbers.
    This guarantees that the exact same random numbers are used in every run.
    """

    def __init__(self, filename="random_normal_table.json"):
        """Create new instance

        :arg filename: name of file to read from.
        """
        with open(filename, "r") as f:
            self.data = np.asarray(json.load(f), dtype=np.float32)
        self.reset()

    def reset(self):
        """Reset pointer to beginning of file"""
        self.ptr = 0

    def take(self, n_samples=1):
        """Take a fixed number of random numbers and returns them as a numpy array

        :arg n_samples: number of samples to take
        """
        self.ptr += n_samples
        if self.ptr >= self.data.size:
            raise RunTimeError("Lookup table exhausted")
        return self.data[self.ptr - n_samples : self.ptr]


class XYModelRandomInitializer:
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


class XYModelConstantInitializer:
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


class RelativisticChargedParticleRandomInitializer:
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


class SingleParticleConstantInitializer:
    """Constant initialiser class for the double well potential model

    :arg dim: dimension of system
    """

    def __init__(self, dim):
        self.dim = dim

    def draw(self):
        """Draw a new sample with

        q_j = 0
        qdot_j = j/d
        """
        rng_table = NormalRandomLookup()
        return rng_table.take(self.dim), rng_table.take(self.dim)


class TwoParticleConstantInitializer:
    """Constant initialiser class for interacting two particle system

    :arg dim: dimension of phase space
    :arg mass1: mass of first particle
    :arg mass2: mass of second particle
    :arg perturbation: size of perturbation to be added to initial vector
    """

    def __init__(self, dim, mass1=1.0, mass2=1.0, perturbation=0):
        self.dim = dim
        assert dim <= 8, "only dimensions up to 8 are supported"
        self.mass1 = mass1
        self.mass2 = mass2
        self.perturbation = perturbation

    def draw(self):
        """Draw a new sample"""
        rng_table = NormalRandomLookup()
        r = rng_table.take(self.dim // 2)
        u1 = list(r / self.mass1)
        u2 = list(-r / self.mass2)
        q = rng_table.take(self.dim) + self.perturbation * rng_table.take(self.dim)
        qdot = u1 + u2 + self.perturbation * rng_table.take(self.dim)
        return (q, qdot)


class KeplerInitializer:
    """Constant initializer for motion in 1/r potential

    def __init__(self, kepler_solution):
        self.kepler_solution = kepler_solution

    def draw(self):
        """Draw a new sample from the exact solution of the Kepler problem"""
        phi = 0.0
        q = self.kepler_solution.position(phi)
        qdot = self.kepler_solution.velocity(phi)
        return q, qdot
