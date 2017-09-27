"""Wrapper for different methods to estimate parameters using homographies."""

from utils.pyslam_object import PyslamObject
from homography_based_methods.simple_recovery import simple_recovery
from homography_based_methods.wadenback_recovery import wadenback_recovery


class ParameterRecovery(PyslamObject):

    def __init__(self, H, method='simple', params={}):
        """Initialize ParameterRecovery."""
        super(ParameterRecovery, self).__init__()

        self.homographies = H
        self.params = params
        self.method = self._check_method(method)

    def _check_method(self, method):
        """Check if method is available and return if so."""
        if method == 'simple':
            return simple_recovery
        elif method == 'wadenback':
            return lambda H: wadenback_recovery(H,
                                                nbr_homographies=self.params.get(
                                                    'nbr_homographies', 'all'),
                                                nbr_iterations=self.params.get('nbr_iterations', 5))
        else:
            msg = "The method '{}' is not implemented.".format(method)
            raise NotImplementedError(msg)

    def reconstruct(self):
        """Path reconstruction using specified method."""
        return self.method(self.homographies)
