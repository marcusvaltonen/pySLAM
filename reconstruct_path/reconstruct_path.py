#!/usr/bin/env python

"""Script to reconstruct path from a sequence of images by first
   precomputing the inter-image homographies. Only homographies
   between two consecutive images are considered."""

from utils.pyslam_object import PyslamObject
from utils.imaging_utils import pairwise, rotm2
from homography.homography import Homography
from parameter_recovery.parameter_recovery import ParameterRecovery

import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class PathReconstructor(PyslamObject):

    def __init__(self, img_path):
        """Initialize PathReconstructor."""
        super(PathReconstructor, self).__init__()

        # Load image meta-data
        self._load_data(img_path)

        # Compute inter-image homographies and store the result
        self.hom = Homography()
        self.homographies = self.precompute_homographies()

    def _load_data(self, img_path):
        """Save path to all images, but do not load them into memory."""
        self.imgs = glob.glob(os.path.join(img_path, '*image*'))
        self.imgs.sort()

        # Load ground truth
        self.ground_truth = np.load(os.path.join(img_path, 'ground_truth.npy'))

    def precompute_homographies(self):
        """Precompute homographies between all consecutive images."""
        return [self.hom.get_homography(im1, im2) for
                (im1, im2) in pairwise(self.imgs)]

    def reconstruct(self, method='simple', params={}):
        """Path reconstruction using specified method."""
        parameter_recovery = ParameterRecovery(self.homographies,
                                               method=method,
                                               params=params)
        self.estimates = parameter_recovery.reconstruct()

        self.path = self._reconstruct_using_estimates(self.estimates)

    def _reconstruct_using_estimates(self, estimates):
        """Reconstruct path using estimated parameters."""
        # Get parameters (for now psi and theta are discarded)
        path = np.hstack((estimates['t'], estimates['phi']))

        # Let first measurement be all zeroes
        path = np.vstack((np.zeros((1, 3)), path))

        # Account for rotation
        phivals = np.cumsum(np.vstack((0, estimates['phi'])))[:-1]
        tvals = estimates['t']
        new_coords = np.zeros_like(tvals)

        for j, (t, phi) in enumerate(zip(tvals, phivals)):
            new_coords[j, :] = rotm2(phi).dot(np.atleast_2d(t).T).flatten()

        # Stack with rotation and pad first value
        path = np.hstack((new_coords, estimates['phi']))
        path = np.vstack((np.zeros((1, 3)), path))

        # Cumulative sum to get the absolute position and rotation
        path = np.cumsum(path, axis=0)

        # Align first point to ground truth for better visualization
        path += self.ground_truth[0, :]

        return path

    def plot_results(self):
        """Plot the results."""
        plt.figure()
        plt.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], 'bo')
        plt.plot(self.path[:, 0], self.path[:, 1], 'r*-')
        plt.gca().invert_yaxis()

        f, axes = plt.subplots(3, 1, sharex=True)

        # Call rad2deg on degrees
        self.path[:, 2] = np.rad2deg(self.path[:, 2])

        # Add labels for nicer and more explanatory plots
        comments = [{'ylabel': r'Error in $x$'},
                    {'ylabel': r'Error in $y$'},
                    {'ylabel': r'Error in $\theta$ (degrees)'}]

        for ax, pat, gth, com in zip(axes, self.path.T,
                                     self.ground_truth.T, comments):
            ax.plot(pat - gth, 'ro-')
            ax.plot(np.zeros_like(pat), 'o',
                    markerfacecolor='none', markeredgecolor='b')
            ax.set_ylim([-5, 5])
            ax.set_ylabel(com['ylabel'])

        plt.xlabel(r'Homography number')

        plt.show()


if __name__ == '__main__':
    img_path = '../generate_path/output'
    path_rec = PathReconstructor(img_path)
    path_rec.reconstruct(method='wadenback',
                         params={'nbr_homographies': 5})
    path_rec.plot_results()
