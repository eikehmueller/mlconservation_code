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
    """Random initialiser class for the XY model"""

    def __init__(self, dim):
        self.dim = dim

    def draw(self):
        """Draw a new sample with

        q_j ~ Uniform(-pi,+pi)
        qdot_j ~ Normal(0,1)
        """
        q = np.zeros(self.dim)
        qdot = np.arange(0, self.dim) / self.dim
        return q, qdot
