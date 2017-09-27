"""Parameter recovery using methods proposed by Wadenback et al."""

from utils.imaging_utils import qr, rotm, merge_dicts
import numpy as np
import random


def wadenback_recovery(H, nbr_homographies='all', nbr_iterations=5):
    """Parameter recovery based on one or several homographies
       using algorithms proposed in Lic. thesis of Marten Wadenback.
    """
    if nbr_homographies == 'all':
        return _wadenback_seq(H, nbr_iterations=nbr_iterations)
    elif isinstance(nbr_homographies, int) and nbr_homographies > 0:

        # Pre-allocate output dictionary
        N = len(H)
        output = {'psi': np.zeros((N, 1)),
                  'theta': np.zeros((N, 1)),
                  'phi': np.zeros((N, 1)),
                  't': np.zeros((N, 2))}

        for j, h in enumerate(H):
            # TODO: How to pick more than one homography? Consecuutive
            # or random? If consecutive, how to deal with boundaries?
            # For now, pick randomly.

            # Generate random indeces
            H_idx = range(N)
            H_idx.remove(j)
            H_idx = random.sample(H_idx, nbr_homographies - 1)
            H_idx.insert(0, j)

            tmp = _wadenback_seq([H[idx] for idx in H_idx],
                                 nbr_iterations=nbr_iterations)
            output['psi'][j] = tmp['psi'][0]
            output['theta'][j] = tmp['theta'][0]
            output['phi'][j] = tmp['phi'][0]
            output['t'][j, :] = tmp['t'][0]

        return output
    else:
        msg = "nbr_homographies should be 'all' or a positive integer"
        raise RuntimeError(msg)


def _wadenback_seq(H, nbr_iterations):
    """Estimate psi, theta, phi and t from a sequence of homographies."""

    M = [np.dot(h.T, h) for h in H]

    R = np.eye(3, 3)
    Rtheta = np.eye(3, 3)

    for j in range(nbr_iterations):

        # Estimate psi
        M = [np.dot(Rtheta.T, np.dot(M[k], Rtheta)) for k in range(len(M))]
        A = [np.array(
            [[m[0, 0] - m[1, 1], -2.0 * m[1, 2], m[0, 0] - m[2, 2]],
             [m[0, 1], m[0, 2], 0.],
             [0., m[0, 1], m[0, 2]]]) for m in M]
        V = np.linalg.svd(reduce(
            lambda x, y: np.append(x, y, axis=0), A))[2].T
        A = reduce(lambda x, y: np.append(x, y, axis=0), A)
        B = A.copy()
        B[:, 0] = A[:, 0] + A[:, 2]
        B[:, 1] = 2 * A[:, 1]
        B[:, 2] = A[:, 0] - A[:, 2]
        V = V[:, 2].copy() / V[0, 2]
        if not all(np.isreal(V)):
            print 'V was not real! (for psi)'
            V = V.real
        if abs(2. * V[1]) > 1:
            print 'abs > 1 (for psi)'
            print V
            psi = np.arcsin(np.sign(2. * V[1])) / 2.
        else:
            psi = np.arcsin(2. * V[1]) / 2.
        Rpsi = rotm(np.array([[1., 0., 0.]]).T, psi)

        # Estimate theta
        M = [np.dot(Rpsi.T, np.dot(M[k], Rpsi)) for k in range(len(M))]
        A = [np.array(
            [[m[0, 0] - m[1, 1], -2.0 * m[0, 2], m[2, 2] - m[1, 1]],
             [m[0, 1], -m[1, 2], 0.],
             [0., m[0, 1], -m[1, 2]]]) for m in M]
        A = reduce(lambda x, y: np.append(x, y, axis=0), A)
        B = A.copy()
        B[:, 0] = A[:, 0] + A[:, 2]
        B[:, 1] = 2 * A[:, 1]
        B[:, 2] = A[:, 0] - A[:, 2]
        V = np.linalg.svd(B)[2].T
        V = V[:, 2].copy() / V[0, 2]
        if not all(np.isreal(V)):
            print 'V was not real! (for theta)'
            V = V.real
        if abs(2. * V[1]) > 1:
            print 'abs > 1 (for theta)'
            print V
            theta = np.arcsin(np.sign(2. * V[1])) / 2.
        else:
            theta = np.arcsin(2. * V[1]) / 2.
        Rtheta = rotm(np.array([[0., 1., 0.]]).T, theta)

        # Update R
        R = np.dot(R, np.dot(Rpsi, Rtheta))

    # Determine the parameters
    N = len(H)
    theta = np.arcsin(R[0, 2]) * np.ones((N, 1))
    psi = np.arcsin(R[2, 1]) * np.ones((N, 1))
    phi = np.zeros((N, 1))
    t = np.zeros((N, 2))

    for j, h in enumerate(H):
        (R_tmp, T_tmp) = qr(np.dot(np.dot(R.T, h), R))
        # Follow sign convention from OpenCV
        phi[j] = np.arcsin(R_tmp[0, 1])
        t[j, :] = -T_tmp[:2, 2]

    return {'psi': psi, 'theta': theta, 'phi': phi, 't': t}
