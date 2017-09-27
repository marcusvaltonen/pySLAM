#!/usr/bin/env python

"""Module with image processing utilities."""

import numpy as np
from itertools import tee
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    import Image


def qr(A):
    """QR decomposition with positive element on diagonal of R."""
    q, r = np.linalg.qr(A)
    sgn = np.sign(np.diag(r))[:, np.newaxis]
    r = r * sgn
    q = q * sgn.T
    return q, r


def pflat(x):
    """Normalize with the last element in each column of an array."""
    m, n = x.shape
    return x / np.tile(x[-1, :], (m, 1))


def crossm(v):
    """Creates a matrix C such that C*x = cross(v, x) for all x."""
    return np.cross(np.tile(v, (1, 3)), np.eye(3, 3), axis=0)


def rotm2(alpha):
    """Create a 2D rotation matrix."""
    alpha = np.asscalar(alpha)
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]])


def rotm(v, alpha):
    """Uses Rodrigues' formula to create a 3D rotation
       matrix rotating the angle alpha around the axis v."""
    W = crossm(v / np.linalg.norm(v))
    return np.eye(3, 3) + np.sin(alpha) * W + \
        2.0 * (np.sin(alpha / 2.0) ** 2) * np.dot(W, W)


def load_image(infilename):
    """Load image as numpy array."""
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype='uint8')
    return data


def save_image(npdata, outfilename):
    """Save numpy array as image."""
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255),
                                     dtype='np.uint8'), 'L')
    img.save(outfilename)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    for elem in b:
        break
    return zip(a, b)


def merge_dicts(dicts, dtype=list):
    """Merge dicts that have common elements."""
    dd = defaultdict(dtype)

    for d in dicts:
        for key, value in d.iteritems():
            dd[key].append(value)

    return dict(dd)
