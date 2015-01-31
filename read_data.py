__author__ = 'dudevil'

import os
import glob
import json
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.cross_validation import StratifiedShuffleSplit


class DataSetLoader:

    def __init__(self, data_dir="data", img_size=48):
        self.data_dir = data_dir
        self.class_labels = {}
        self._num_label = 0
        self.img_size = img_size
        self.train_file = os.path.join("data", "tidy", "train_%d.npy" % img_size)
        self.trainlabels_file = os.path.join("data", "tidy", "train_labels_%d.npy" % img_size)
        self.test_file = os.path.join("data", "tidy", "test_%d.npy" % img_size)
        self.n_testimages = 130400
        # filenames in the testset order
        self.testfilenames = []

    def load_train(self):
        # check if a dataset with the given image size has already been processed
        if os.path.isfile(self.train_file) and os.path.isfile(self.trainlabels_file):
            X = np.load(self.train_file)
            y = np.load(self.trainlabels_file)
            with open(os.path.join("data", "tidy", "train_%d_labmapping.npy" % self.img_size), 'r') as lfile:
                self.class_labels = json.load(lfile)
            return X, y
        x = []
        y = []
        for directory in sorted(glob.iglob(os.path.join(self.data_dir, "train", "*"))):
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
        # save the processed files
        np.save(self.train_file, x)
        np.save(self.trainlabels_file, y)
        # also save label to index mapping
        with open(os.path.join("data", "tidy", "train_%d_labmapping.npy" % self.img_size), 'w') as lfile:
            json.dump(self.class_labels, lfile, indent=2)
        return x, y

    def load_test(self):
        testdir = os.path.join(self.data_dir, "test")
        # if a test dataset is present load it from file
        if os.path.isfile(self.test_file):
            self.testfilenames = os.listdir(testdir)
            return np.load(self.test_file)
        # read test images
        images = np.zeros((self.n_testimages, self.img_size * self.img_size), dtype='float32')
        for i, imfile in enumerate(os.listdir(testdir)):
            img_array = imread(os.path.join(testdir, imfile), as_grey=True)
            images[i, ...] = resize(img_array, (self.img_size, self.img_size)).reshape(1, -1)
            self.testfilenames.append(imfile)
        assert len(images) == len(self.testfilenames), "Number of files doesn't match number of images"
        # cache the resulting array for future use
        np.save(self.test_file, images)
        return images

    def train_test_split(self, test_size=0.1, random_state=0):
        X, y = self.load_train()
        sss = StratifiedShuffleSplit(y, n_iter=1, random_state=random_state, test_size=test_size)
        # we only split once so do not use iter, just convert to list and get first split
        train, test = list(sss).pop()
        return X[train], X[test], y[train], y[test]

    def save_submission(self, y_pred, file_suffix=""):
        # sanity-check
        h, w = y_pred.shape
        assert w == len(self.class_labels), "Not all class labels present"
        # number of test cases
        assert h == len(self.testfilenames), "Not all test observations present"
        colnames = [self.class_labels[str(ind)] for ind in xrange(121)]
        dfr = pd.DataFrame(y_pred, index=self.testfilenames, columns=colnames)
        dfr.to_csv(os.path.join(self.data_dir, "submissions", "submission-%s.csv" % file_suffix),
                   format="%f",
                   index_label="image")
