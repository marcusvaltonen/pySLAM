#!/usr/bin/env python

"""Script to recover parameters using the method proposed
   by Wadenback and Heyden in 'Ego-Motion Recovery and
   Robust Tilt Estimation for Planar Motion Using Several
   Homographies'. The plots that are generated are supposed
   to mimick those presented in the paper.
"""

from utils.pyslam_object import PyslamObject
from utils.imaging_utils import rotm
from parameter_recovery import ParameterRecovery

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


class HomographyAnalyzer(PyslamObject):

    def __init__(self):
        """Initialize HomographyAnalyzer."""
        super(HomographyAnalyzer, self).__init__()

    def generate_homographies(self, N=50, sigma=0, psi=None, theta=None):
        """Generate homographies."""
        psi = psi * \
            np.ones(N) if psi is not None else np.deg2rad(
                5. * np.random.randn(N))
        theta = theta * \
            np.ones(N) if theta is not None else np.deg2rad(
                5. * np.random.randn(N))
        phi = np.deg2rad(55. * np.random.rand(N) + 15.)
        t = 6. * np.random.rand(2, N) - 3.

        # Generate homograpies
        H = []

        for j in range(N):
            Rpsi = rotm(np.array([[1., 0., 0.]]).T, psi[j])
            Rtheta = rotm(np.array([[0., 1., 0.]]).T, theta[j])
            # Sign convention from OpenCV
            Rphi = rotm(np.array([[0., 0., 1.]]).T, -phi[j])
            Rpt = np.dot(Rpsi, Rtheta)
            T = np.eye(3, 3)
            T[:2, 2] = -t[:, j]
            H.append(np.dot(np.dot(Rpt, Rphi), np.dot(T, Rpt.T)))

        self.homographies = H
        self.ground_truth = {'psi': psi, 'theta': theta, 'phi': phi, 't': t}

    def recover_parameters(self, method='simple', params={}):
        """Recover parameters from inter-image homographies."""
        parameter_recovery = ParameterRecovery(self.homographies,
                                               method=method,
                                               params=params)
        self.estimates = parameter_recovery.reconstruct()

    def plot_results(self):
        """Plot results."""
        # Get ground truth
        psi = self.ground_truth['psi']
        theta = self.ground_truth['theta']
        phi = self.ground_truth['phi']
        t = self.ground_truth['t']

        # Get estimates
        psi_est = self.estimates['psi']
        theta_est = self.estimates['theta']
        phi_est = self.estimates['phi']
        t_est = self.estimates['t']

        # Use LaTeX fonts
        rc('text', usetex=True)
        rc('font', **dict(family='serif', serif='computer modern roman'))
        plt.figure()

        plt.subplot(4, 1, 1)
        plt.plot(np.rad2deg(psi_est), 'r-o', mec='r')
        plt.plot(np.rad2deg(psi), 'bo', mec='b', mfc='None', mew=2)
        plt.xlabel(r'Homography number')
        plt.ylabel(r'$\psi$ (degrees)')
        plt.ylim([-2.2, -1.2])
        plt.yticks(np.arange(-2.2, -1.1, 0.2))

        plt.subplot(4, 1, 2)
        plt.plot(np.rad2deg(theta_est), 'r-o', mec='r')
        plt.plot(np.rad2deg(theta), 'bo', mec='b', mfc='None', mew=2)
        plt.xlabel(r'Homography number')
        plt.ylabel(r'$\theta$ (degrees)')
        plt.ylim([4.6, 5.6])
        plt.yticks(np.arange(4.6, 5.7, 0.2))

        plt.subplot(4, 1, 3)
        plt.plot(np.rad2deg(phi_est), 'r-o', mec='r')
        plt.plot(np.rad2deg(phi), 'bo', mec='b', mfc='None', mew=2)
        plt.xlabel(r'Homography number')
        plt.ylabel(r'$\varphi$ (degrees)')
        plt.ylim([10, 75])
        plt.yticks(np.arange(15, 76, 15))

        plt.subplot(4, 1, 4)
        plt.plot(np.zeros_like(t_est[:, 0]), 'bo', mec='b')
        plt.plot(t_est[:, 1] - t[1, :], 'r-o', mec='r')
        plt.plot(t_est[:, 0] - t[0, :], 'g--d', mec='g')
        plt.xlabel(r'Homography number')
        plt.ylabel(r'Error in $t_x$ and $t_y$')
        plt.ylim([-0.1, 0.1])
        plt.yticks(np.arange(-0.1, 0.11, 0.05))

        plt.tight_layout()


if __name__ == '__main__':
    ha = HomographyAnalyzer()
    ha.generate_homographies(psi=np.deg2rad(-1.56), theta=np.deg2rad(5.23))
    ha.recover_parameters(method='wadenback', params={
                          'nbr_homographies': 1})
    ha.plot_results()
    ha.recover_parameters(method='wadenback', params={
                          'nbr_homographies': 5})
    ha.plot_results()
    plt.show()
