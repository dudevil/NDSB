__author__ = 'dudevil'

import skimage.transform
import numpy as np

def square(image, resize=(48, 48), flatten=True):
    img_x, img_y = image.shape
    diff = np.abs(img_y - img_x)
    # pad image from two sides, if necessary add extra pixel at one side
    pad = (diff/2 + diff % 2, diff / 2)
    padding = [(0, 0), (0, 0)]
    # only pad the smaller axis
    padding[np.argmin(image.shape)] = pad
    padded = np.pad(image, padding, mode='constant', constant_values=(255,))
    if resize and flatten:
        return skimage.transform.resize(padded, resize).flatten()
    if resize:
        return skimage.transform.resize(padded, resize)
    if flatten:
        return padded.flatten()
    return padded


def transform(image, max_angle=360, flip=False, shift=4, zoom_range=(1/1.2, 1.2),
              image_size=(48,48),
              flatten=True, rng=np.random.RandomState(123)):

    if max_angle:
        # rotate an image to a random angle
        rot_angle = rng.randint(0, max_angle)
        # 50 % chance of flipping the image
        if flip and rng.randint(2) > 1.:
            rot_angle += 180
    else:
        rot_angle = 0
    if shift:
        # shift an image randomly in (-shift, shift) boundaries
        shift_x = rng.randint(-shift, shift)
        shift_y = rng.randint(-shift, shift)
    else:
        shift_x = 0
        shift_y = 0

    if zoom_range:
        # zoom by a factor of log-uniform from zoom_range
        log_zoom_range = np.log(zoom_range)
        zoom = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom = 1.

    # we need to rotate around the center, so we first shift the image and then rotate
    center_y, center_x = np.array(image.shape[:2]) / 2.
    rotate = skimage.transform.SimilarityTransform(rotation=np.deg2rad(rot_angle))
    center_shift = skimage.transform.SimilarityTransform(translation=[-center_x, -center_y])
    # distort backshifting by shift factors
    shift_inv = skimage.transform.SimilarityTransform(translation=[center_x + shift_x, center_y + shift_y])
    zoom = skimage.transform.SimilarityTransform(scale=zoom)
    res = skimage.transform.warp(image, (center_shift + (rotate + zoom + shift_inv)), mode='nearest', output_shape=image_size)
    if flatten:
        return res.reshape(1, -1)
    return res