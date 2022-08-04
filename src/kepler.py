"""Class for analytical solution of motial in a 1/r potential"""
import numpy as np


class KeplerSolution(object):
    """Class representing an exact solution of motion in a central 1/r potential

    The potential is given by -alpha/r, and the solutions for negative energy are
    ellipses with a given excentricity.

    The motion is three-dimensional but confined to the x-y plane.

    :arg mass: particle mass
    :arg alpha: strength of potential
    :arg excentricity: excentricity of the trajectory
    :arg energy: energy (must be negative)
    :arg phi0: initial angle
    """

    def __init__(self, mass=1.0, alpha=1.0, excentricity=0.0, energy=-1.0, phi0=0.0):
        self.mass = mass
        self.alpha = alpha
        self.excentricity = excentricity
        self.energy = energy
        self.phi0 = phi0
        self.L_angular = self.alpha * np.sqrt(
            -0.5 * self.mass / self.energy * (1 - self.excentricity**2)
        )

    def _u(self, phi):
        """Compute the function u(phi) = 1/r(phi) for a given angle phi

        :arg phi: angle phi
        """
        return (
            self.alpha
            * self.mass
            / self.L_angular**2
            * (1 + self.excentricity * np.cos(phi - self.phi0))
        )

    def position(self, phi):
        """Return 3d position vector as function of the angle phi

        :arg phi: angle
        """
        phi = np.asarray(phi)
        r = (
            self.L_angular**2
            / (self.alpha * self.mass)
            * 1
            / (1 + self.excentricity * np.cos(phi - self.phi0))
        )
        return np.stack(
            [r * np.cos(phi), r * np.sin(phi), np.zeros(shape=phi.shape)], axis=0
        )

    def velocity(self, phi):
        """Return 3d velocity vector as function of the angle phi

        :arg phi: angle
        """
        phi = np.asarray(phi)
        u = self._u(phi)
        r = 1 / u
        rdot = self.alpha / self.L_angular * self.excentricity * np.sin(phi - self.phi0)
        r_phidot = self.L_angular / self.mass * u
        return np.stack(
            [
                rdot * np.cos(phi) - r_phidot * np.sin(phi),
                rdot * np.sin(phi) + r_phidot * np.cos(phi),
                np.zeros(shape=phi.shape),
            ],
            axis=0,
        )

    def acceleration(self, phi):
        """Return 3d acceleration vector as function of the angle phi

        :arg phi: angle
        """
        phi = np.asarray(phi)
        u = self._u(phi)
        r = 1 / u
        rddot = (
            self.alpha
            / self.mass
            * self.excentricity
            * u**2
            * np.cos(phi - self.phi0)
        )
        phidot = self.L_angular / self.mass * u**2
        acc = rddot - r * phidot**2
        return np.stack(
            [acc * np.cos(phi), acc * np.sin(phi), np.zeros(shape=phi.shape)], axis=0
        )
