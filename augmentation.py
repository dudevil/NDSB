__author__ = 'dudevil'

from multiprocessing import Process
#import prctl
from sys import platform as _platform
from signal import SIGHUP
from Queue import Full
import skimage.transform
import numpy as np
# import cProfile
#
# def do_cprofile(func):
#     def profiled_func(*args, **kwargs):
#         profile = cProfile.Profile()
#         try:
#             profile.enable()
#             result = func(*args, **kwargs)
#             profile.disable()
#             return result
#         finally:
#             profile.print_stats()
#     return profiled_func

def square(image, output_shape=(48, 48), flatten=True):
    img_x, img_y = image.shape
    diff = np.abs(img_y - img_x)
    # pad image from two sides, if necessary add extra pixel at one side
    pad = (diff/2 + diff % 2, diff / 2)
    padding = [(0, 0), (0, 0)]
    # only pad the smaller axis
    padding[np.argmin(image.shape)] = pad
    padded = np.pad(image, padding, mode='constant', constant_values=(255,))
    if output_shape and flatten:
        return skimage.transform.resize(padded, output_shape).flatten()
    if output_shape:
        return skimage.transform.resize(padded, output_shape)
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

class Augmenter(Process):

    def __init__(self,
                 queue,
                 images,
                 max_items,
                 random_seed=0,
                 max_angle=360,
                 max_shift=4,
                 normalize=True,
                 flatten=True):
        super(Augmenter, self).__init__()
        self.q = queue
        self.rng = np.random.RandomState(random_seed)
        self.max_angle = max_angle
        self.flatten = flatten
        self.images = images
        self.items_count = max_items
        self.shift = max_shift
        self.zoom_range = (1/1.1, 1.1)
        self.normalize = normalize
        self.angle = 0
        # kill this process if parent process dies
        # this only works on linux, so you should update the code or kill the process
        # yourself if using other OSes
        # if _platform == "linux" or _platform == "linux2":
        #     prctl.set_pdeathsig(SIGHUP)

    def rand_rotate(self, image):
        angle = self.rng.randint(0, self.max_angle)
        shift_x = self.rng.randint(-self.shift, self.shift)
        shift_y = self.rng.randint(-self.shift, self.shift)
        log_zoom_range = np.log(self.zoom_range)
        #zoom = np.exp(self.rng.uniform(*log_zoom_range))
        #output = skimage.transform.rotate(image, angle,  mode='constant', cval=1.0)
        center_y, center_x = np.array(image.shape[:2]) / 2.
        rotate = skimage.transform.SimilarityTransform(rotation=np.deg2rad(angle))
        center_shift = skimage.transform.SimilarityTransform(translation=[-center_x, -center_y])
        # distort backshifting by shift factors
        shift_inv = skimage.transform.SimilarityTransform(translation=[center_x, center_y])
        shift = skimage.transform.SimilarityTransform(translation=[shift_x, shift_y])
        #zoom = skimage.transform.SimilarityTransform(scale=zoom)
        output = skimage.transform.warp(image, (center_shift + (rotate + shift_inv) + shift),
                                        mode='constant', cval=1.0)
        if self.flatten:
            return output.flatten()
        return output

    def rotate90(self, image):
        center_y, center_x = np.array(image.shape[:2]) / 2.
        shift_x = self.rng.randint(-self.shift, self.shift)
        shift_y = self.rng.randint(-self.shift, self.shift)
        rotate = skimage.transform.SimilarityTransform(rotation=np.deg2rad(self.angle))
        center_shift = skimage.transform.SimilarityTransform(translation=[-center_x, -center_y])
        # distort backshifting by shift factors
        shift_inv = skimage.transform.SimilarityTransform(translation=[center_x, center_y])
        shift = skimage.transform.SimilarityTransform(translation=[shift_x, shift_y])
        #zoom = skimage.transform.SimilarityTransform(scale=zoom)
        output = skimage.transform.warp(image, (center_shift + (rotate + shift_inv) + shift),
                                        mode='constant', cval=1.0)
        self.angle = (self.angle + 15) % 360
        if self.flatten:
            return output.flatten()
        return output


#   @do_cprofile
    def run(self):
        try:
            while self.items_count:
                rotated = np.vstack(tuple(map(self.rotate90, self.images)))
                self.q.put(rotated, block=True, timeout=360)
                self.items_count -= 1
        except Full:
        # probably the consumer is not processing values anymore
            print("Queue full")
        print("Exiting")
