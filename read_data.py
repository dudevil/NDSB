__author__ = 'dudevil'

import os
import glob
import numpy as np
from skimage.io import imread
from skimage.transform import resize


class DataSetLoader:

    def __init__(self, data_dir="data", img_size=48):
        self.data_dir = data_dir
        self.class_labels = {}
        self._num_label = 0
        self.img_size = img_size
        self.train_file = os.path.join("data", "tidy", "train_%d.npy" % img_size)
        self.test_file = os.path.join("data", "tidy", "test_%d.npy" % img_size)

    def load_train(self):
        # check if a dataset with the given width has already benn processed
        if os.path.isfile(self.train_file):
            data = np.load(self.train_file)
            X = data[:, :-1]
            y = data[:, -1]
            y = y.astype('int32')
            return X, y
        x = []
        y = []
        for directory in glob.iglob(os.path.join(self.data_dir, "train", "*")):
            print("processing %s" % directory)
            files = os.listdir(directory)
            # set up the array to store images and labels
            n_images = len(files)
            images = np.zeros((n_images, self.img_size * self.img_size), dtype='float32')
            # the last directory is a class label
            self.class_labels[self._num_label] = os.path.split(directory)[-1]
            # create labels list
            y.extend([self._num_label] * n_images)
            self._num_label += 1
            for i, image in enumerate(files):
                img_array = imread(os.path.join(directory, image), as_grey=True)
                images[i, ...] = resize(img_array, (self.img_size, self.img_size)).reshape(1, -1)
            x.append(images)
        # concatenate the arrays from all classes and append labels
        x = np.vstack(tuple(x))
        y = np.array(y, dtype='int32')
        # save the processed file
        np.save(self.train_file, np.hstack((x, y.reshape(-1, 1))))
        return x, y