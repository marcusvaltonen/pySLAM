"""Base class for pySLAM objects."""

import numpy as np
from matplotlib import rc


class PyslamObject(object):

    def __init__(self):
        """Initialize all common parameters."""
        rc('text', usetex=True)
        rc('font', **dict(family='serif', serif='computer modern roman'))
        np.set_printoptions(suppress=True, precision=8, linewidth=160)
