import random

import numpy as np

from .elastic_deform import ElasticTransform
from .random_crop import RandomCropToLabels
from .random_flip import RandomFlip
from .random_rescale import RandomZoom
from .random_rotate import RandomRotation, StaticRotation
from .random_shift import RandomShift
from .gaussian_noise import GaussianNoise

functions = ['elastic_deform', 'random_crop', 'random_flip', 'random_rescale', 'random_rotate', 'random_shift']


'''
# the transform function take as import image of size CDHW numpy format, label of size DHW numpy format, 
# and output image of numpy format, label of numpy format.
'''


class RandomChoice(object):
    """
    choose a random tranform from list an apply
    transforms: tranforms to apply
    p: probability
    """

    def __init__(self, transforms=[],
                 p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors, label
        t = random.choice(self.transforms)

        for i in range(len(img_tensors)):

            if i == (len(img_tensors) - 1):
                ### do only once the augmentation to the label
                img_tensors[i], label = t(img_tensors[i], label)
            else:
                img_tensors[i], _ = t(img_tensors[i], label)
        return img_tensors, label


class ComposeTransforms(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms=[],
                 p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        # print('here1: ', img_tensors.shape)
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors, label
        # print('here2: ', img_tensors.shape)
        for i in range(len(img_tensors)):
            # print('here3: ', img_tensors.shape)
            for t in self.transforms:
                if i == (len(img_tensors) - 1):
                    ### do only once augmentation to the label
                    # print('here4: ', img_tensors.shape)
                    img_tensors[i], label = t(img_tensors[i], label)
                    # print('here5: ', img_tensors.shape)
                else:
                    # print('here6: ', img_tensors.shape)
                    img_tensors[i], _ = t(img_tensors[i], label)
                    # print('here7: ', img_tensors.shape)
        return img_tensors, label
