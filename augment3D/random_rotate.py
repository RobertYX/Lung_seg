import random
import numpy as np
import scipy.ndimage as ndimage


def random_rotate3D(img_numpy, min_angle, max_angle):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return: 3D rotated img
    """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes, reshape=False, mode='nearest', order=0)

def static_rotate3D(img_numpy, all_angle):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return: 3D rotated img
    """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    all_axes = [(1, 0), (1, 2), (0, 2)]
    if len(all_angle)==0:
        all_angle = [-270, -180, -90, 90, 180, 270]
    angle = random.choice(all_angle)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes, reshape=False, mode='nearest', order=0)

class RandomRotation(object):
    def __init__(self, min_angle=-10, max_angle=10):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated

        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        img_numpy = random_rotate3D(img_numpy, self.min_angle, self.max_angle)
        if label.any() != None:
            label = random_rotate3D(label, self.min_angle, self.max_angle)
        return img_numpy, label

class StaticRotation(object):
    def __init__(self, all_angel=[-270, -180, -90, 90, 180, 270]):
        self.all_angle = all_angel

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated

        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        img_numpy = static_rotate3D(img_numpy, self.all_angle)
        if label.any() != None:
            label = static_rotate3D(label, self.all_angle)
        return img_numpy, label