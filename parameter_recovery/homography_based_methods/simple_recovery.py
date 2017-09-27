"""Simple method for recovering parameters from homographies."""

from utils.imaging_utils import qr

import numpy as np


def simple_recovery(H):
    """Basic reconstruction based on a single homography."""

    N = len(H)
    estimates = {'psi': np.zeros((N, 1)),
                 'theta': np.zeros((N, 1)),
                 'phi': np.zeros((N, 1)),
                 't': np.zeros((N, 2))}
    # TODO: Make _simple_recovery return something nicer.
    for j, h in enumerate(H):
        txc, tyc, rot = _simple_recovery(h)
        estimates['psi'][j] = 0
        estimates['theta'][j] = 0
        estimates['phi'][j] = rot
        estimates['t'][j, :] = np.array([txc, tyc])

    return estimates


def _simple_recovery(H):
    """Extract the parameters from homography H."""
    Hinv = np.linalg.inv(H)
    Q, R = qr(Hinv)

    txc = Hinv[0][2]
    tyc = Hinv[1][2]

    rot = np.arctan2(Q[1, 0], Q[0, 0])

    return txc, tyc, rot
