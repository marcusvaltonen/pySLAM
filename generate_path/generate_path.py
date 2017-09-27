#!/usr/bin/env python

"""Generate path in a given image."""

import optparse
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    import Image

import cv2
import os

import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.collections import PatchCollection


class PathGenerator(object):

    _debug = False
    _grid = False
    _force = False

    def __init__(self, im, n):
        '''Initialize PathGenerator.
        N: Total amount of pixels N x N
        n: Pixels per frame (n x n)
        '''
        self.frame_size = n
        self.total_shape = im.shape
        self.frame_shape = (n, n)
        self.im = im
        self._outputdir = ''
        self._prefix = 'image'
        self._noise_std = 0

    def set_path(self, points):
        '''Construct path from coordinates and angles.'''
        self.path = np.array(points)

        self.images = []
        self.frame_info = []

        # Generate images
        for x, y, angle in points:
            im = self.rotate(self.im, (y, x), angle, self.frame_size)
            self.images.append(self.add_noise(im))
            self.frame_info.append(((x, y), self.frame_size, angle))

        np.save(os.path.join(self._outputdir, 'ground_truth.npy'), self.path)

    def set_output(self, path):
        """Set output path for images."""
        self._outputdir = path
        if self._outputdir and not os.path.exists(self._outputdir):
            os.makedirs(self._outputdir)

    def set_prefix(self, prefix):
        """Set prefix for images."""
        self._prefix = unicode(prefix)

    def set_force(self, force):
        """Enable force mode."""
        if not isinstance(force, bool):
            raise RuntimeError("Force mdoe must be a bool.")
        self._force = force

    def set_debug(self, debug):
        """Enable debug mode."""
        if not isinstance(debug, bool):
            raise RuntimeError("Debug must be a bool.")
        self._debug = debug

    def set_grid(self, grid, size=40):
        """Enable a grid."""
        if not isinstance(grid, bool):
            raise RuntimeError("Grid must be a bool.")
        self._grid = grid

        if not isinstance(size, int):
            raise RuntimeError("Grid size must be an int.")
        self._grid_size = size

        # Add grid to self.im
        if self._grid:
            self.im[::self._grid_size, :, :] = 255
            self.im[:, ::self._grid_size, :] = 255

    def set_noise(self, noise):
        """Set std of noise."""
        self._noise = float(noise)

    def set_tracking_point(self, track):
        """Set tracking point."""
        if track in ['center', 'origin']:
            self._tracking_point = track
        else:
            msg = "Tracking point should be 'center' or 'origin'."
            raise RuntimeError(msg)

    def generate_collection(self):
        self.img_rectangles = []

        for xy, n, angle in self.frame_info:
            x, y = xy
            if self._tracking_point == 'origin':
                rect = patches.Rectangle((x, y), n, n)
            if self._tracking_point == 'center':
                rect = patches.Rectangle((x - n / 2, y - n / 2), n, n)
            tran = mpl.transforms.Affine2D().rotate_deg_around(x, y, angle)
            rect.set_transform(tran)
            self.img_rectangles.append(rect)

    def plot(self):
        if not hasattr(self, 'img_rectangles'):
            self.generate_collection()

        if self._debug:
            for i, (rect, im) in enumerate(zip(self.img_rectangles,
                                               self.images)):
                # Create plot
                fig, ax = plt.subplots(1, 2)

                ax[0].imshow(self.im)
                coll = PatchCollection(self.img_rectangles, cmap=mpl.cm.jet,
                                       alpha=0.25, edgecolor='white')
                single_coll = PatchCollection([rect], cmap=mpl.cm.jet,
                                              alpha=0.25, edgecolor='green',
                                              facecolor='green')
                ax[0].add_collection(coll)
                ax[0].add_collection(single_coll)

                ax[0].plot([x[0][0] for x in self.frame_info],
                           [x[0][1] for x in self.frame_info], 'ro-')
                ax[0].set_xlim(0, self.total_shape[1])
                ax[0].set_ylim(self.total_shape[0], 0)

                ax[1].imshow(im)
                fig.suptitle('x0: {}, y0: {}, angle: {}'.format(
                    self.frame_info[i][0][0],
                    self.frame_info[i][0][1],
                    self.frame_info[i][2]))
                fig.set_size_inches(18.5, 10.5)
                fig.savefig(os.path.join(self._outputdir,
                                         'debug_path{0:03d}.png'.format(i)),
                            bbox_inches='tight', dpi=100)
                plt.close(fig)
        else:
            # Create plot
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.imshow(self.im)
            coll = PatchCollection(self.img_rectangles, cmap=mpl.cm.jet,
                                   alpha=0.25, edgecolor='white')
            ax.add_collection(coll)
            ax.plot([x[0][0] for x in self.frame_info],
                    [x[0][1] for x in self.frame_info], 'ro-')
            ax.set_xlim(0, self.total_shape[1])
            ax.set_ylim(0, self.total_shape[0])
            ax.invert_yaxis()
            plt.show()

    def save_images(self, path=None):
        if not hasattr(self, 'images'):
            raise RuntimeError(
                'You need to run "set_path" before "save_images".')

        for i, im in enumerate(self.images):
            pil = Image.fromarray(np.uint8(im))
            fname = os.path.join(self._outputdir,
                                 '{0}{1:03d}.png'.format(self._prefix, i))

            if os.path.exists(fname) and not self._force:
                if not self.ask_user_input(fname):
                    continue
            pil.save(fname)

    def add_noise(self, im):
        '''Put some random noise on top.'''
        noise = np.random.normal(0, self._noise, im.shape)
        im = im + noise
        return np.clip(im, 0, 255).astype(np.uint8)

    def rotate(self, image, center, angle, n):
        """Rotate image and crop out wanted part."""
        # TODO: Only rotate parts of image that we need
        # and not the entire image.
        # Angles should be in degrees
        if self._tracking_point == 'origin':
            # OpenCV return (y,x)
            rot_mat = cv2.getRotationMatrix2D(center[::-1], angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[:2],
                                    flags=cv2.INTER_LINEAR)

            # Cut out the part that we want
            return result[center[0]:center[0] + n,
                          center[1]:center[1] + n, :]
        elif self._tracking_point == 'center':
            # OpenCV return (y,x)
            rot_mat = cv2.getRotationMatrix2D(center[::-1], angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[:2],
                                    flags=cv2.INTER_LINEAR)

            # Cut out the part that we want
            return result[center[0] - n / 2:center[0] + n / 2,
                          center[1] - n / 2:center[1] + n / 2, :]

    def ask_user_input(self, fname):
        """Ask the user if they want to overwrite an existing file."""
        question = 'The file {} already exists. ' \
                   'Do you want to overwrite it? '.format(fname)
        selection = '([y]es/[n]o/[a]lways/[c]ancel): '
        reply = str(raw_input(question + selection)).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'a':
            self._force = True
            return True
        if reply[0] == 'n':
            return False
        if reply[0] == 'c':
            sys.exit(0)
        else:
            return ask_user_input("Please enter (y/n/a/c))")


if __name__ == '__main__':

    np.set_printoptions(linewidth=160, precision=4)

    parser = optparse.OptionParser()
    parser.add_option('-i', '--input', metavar='IMAGE',
                      default='floor.jpeg',
                      help='Path to image file.')
    parser.add_option('-p', '--prefix', metavar='NAME',
                      default='image',
                      help='The name of the image files (prefix)')
    parser.add_option('-o', '--output', metavar='DIR',
                      default='output',
                      help='Store output and logs in DIR. Default output.')
    parser.add_option('-d', '--debug',
                      default=False,
                      help='Enable debug mode',
                      action='store_true')
    parser.add_option('-f', '--force',
                      default=False,
                      help='Overwrite old files',
                      action='store_true')
    parser.add_option('-g', '--grid',
                      default=False,
                      help='Add a grid (useful in combination with debug)',
                      action='store_true')
    parser.add_option('-n', '--noise', metavar='DIR',
                      default=5,
                      help='Standard deviation of added noise per frame.')
    parser.add_option('-s', '--supress',
                      default=False,
                      help='Supress matplotlib plots',
                      action='store_true')
    parser.add_option('-t', '--track', metavar='TRACK',
                      default='origin',
                      help='Which point in the image that should be tracked. '
                           'Options are center and origin. Default is origin.')
    options, args = parser.parse_args()

    im = np.array(Image.open(options.input))

    n = 400
    pg = PathGenerator(im, n)

    # Settings
    pg.set_output(options.output)
    pg.set_prefix(options.prefix)
    pg.set_debug(options.debug)
    pg.set_force(options.force)
    pg.set_grid(options.grid)
    pg.set_noise(options.noise)
    pg.set_tracking_point(options.track)

    # Define a path
    # Ellipse
    N = 48
    t = np.arange(N + 1)
    x = (1200 + float(1.5 * n) * np.cos(2 * np.pi * t / N)).round().astype(int)
    y = (1000 + float(2 * n) * np.sin(2 * np.pi * t / N)).round().astype(int)
    arr = [(xx, yy, 2 * i) for i, (xx, yy) in enumerate(zip(x, y))]

    # Simple path
    """
    x = (700, 800, 800, 900, 1000)
    y = (700, 800, 900, 1000, 1200)
    r = (0, 45, 0, 45, 45)
    arr = [(xx, yy, rr) for xx, yy, rr in zip(x, y, r)]
    """

    # Only rotations
    """
    x = (2000, 2000, 2000, 2000, 2000)
    y = x
    r = (0, 10, 20, 30, 40, 50)
    arr = [(xx, yy, rr) for xx, yy, rr in zip(x, y, r)]
    """
    # Wheel
    """
    x = (419, 419, 419)
    y = (415, 415, 415)
    r = (0, 22.5, 45)
    arr = [(xx, yy, rr) for xx, yy, rr in zip(x, y, r)]
    """

    # Generate
    pg.set_path(arr)
    if not options.supress:
        pg.plot()
    pg.save_images()
