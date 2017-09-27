#!/usr/bin/env python

"""Calculate Homography between two images."""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Homography(object):

    _routine_map = dict()

    def __init__(self, feature_detection='sift', matcher='bf'):
        """Initialize possible routines."""
        self.feature_detection = None
        self.matcher = None
        self._generate_routine_map()

        # Overwrite feature detection and matcher
        self._map_routine(feature_detection)()
        self._map_routine(matcher)()

    def _generate_routine_map(self):
        """Generate routine map."""
        self._routine_map = {'sift': self._setup_sift,
                             'orb': self._setup_orb,
                             'bf': self._setup_bf,
                             'flann': self._setup_flann}

    @property
    def supported_routines(self):
        """List of all supported routines."""
        return self._routine_map.keys()

    def _map_routine(self, routine):
        """Map a routine from string."""
        if routine not in self.supported_routines:
            msg = 'The requested routine {} is not available'.format(routine)
            raise NotImplementedError(msg)

        return self._routine_map[routine]

    def _setup_sift(self):
        """Setup SIFT."""
        self.feature_detection = cv2.xfeatures2d.SIFT_create()

    def _setup_orb(self):
        """Setup ORB."""
        self.feature_detection = cv2.ORB_create()

    def _setup_bf(self):
        """Setup BF."""
        self.matcher = cv2.BFMatcher()

    def _setup_flann(self):
        """Setup FLANN."""
        """ BEWARE! There is a bug in 3.1.0's Python binding of OpenCV.
            Check https://github.com/opencv/opencv/issues/5667 for updates.
            The fix will be available in 3.1.1.
        """
        # Define FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary

        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def get_homography(self, img1, img2, plot=False, rigid=True):
        """Compute a homography between two images"""
        if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
            pass
        elif isinstance(img1, basestring) and isinstance(img2, basestring):
            img1 = cv2.imread(img1)
            img2 = cv2.imread(img2)
        else:
            msg = 'Unsupported type; use numpy.ndarray or string'
            raise RuntimeError(msg)

        img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Find descriptors in each image
        kp1, des1 = self.feature_detection.detectAndCompute(img1gray, None)
        kp2, des2 = self.feature_detection.detectAndCompute(img2gray, None)

        # Match descriptors between images
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        if rigid:
            full_affine = False
            M = cv2.estimateRigidTransform(src_pts, dst_pts, full_affine)
            if M is None:
                raise RuntimeError('Could not estimate rigid transform.')
            M = np.vstack((M, np.eye(1, 3, 2)))
        else:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if plot:
            if not rigid:
                matchesMask = mask.ravel().tolist()

                h, w = img1gray.shape
                pts = np.float32(
                    [[0, 0], [0, h - 1],
                     [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Draw matches in green (only inliers)
                draw_params = dict(matchColor=(0, 255, 0),
                                   singlePointColor=None,
                                   matchesMask=matchesMask,
                                   flags=2)

                img3 = cv2.drawMatches(img1, kp1, img2, kp2, good,
                                       None, **draw_params)

                plt.figure()
                plt.imshow(img3)

            # Draw stitching
            stitch = self.warp_two_images(img1, img2, np.linalg.inv(M))
            plt.imshow(stitch)

            plt.show()
        return M

    def warp_two_images(self, img1, img2, H):
        '''Warp img2 to img1 with homograph H.'''
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        pts1 = np.float32(
            [[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32(
            [[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2_ = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2_), axis=0)
        xmin, ymin = np.int32(pts.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]

        # Generate translation homography
        Ht = np.eye(3)
        Ht[:2, 2] = t

        result = cv2.warpPerspective(img2, Ht.dot(H),
                                     (xmax - xmin, ymax - ymin))
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
        return result


if __name__ == '__main__':
    import os
    import glob

    # For nicer output
    np.set_printoptions(suppress=True, precision=4, linewidth=160)
    hom = Homography()

    img_path = '../generate_path/output'
    imgs = glob.glob(os.path.join(img_path, '*image*'))
    imgs.sort()

    M = hom.get_homography(imgs[0], imgs[1], rigid=False, plot=True)

    print M
