__author__ = 'dudevil'
"""
This module provides an utility class to load images from the training and test sets.
Provides methods for train-validation stratified split.

train_gen and valid_gen methods create a python generator indefinetaly yielding train/ and validation set.

For now only shuffling is performed on the train and validation sets.
"""


import os
import glob
import cPickle
import functools
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.cross_validation import StratifiedShuffleSplit
from augmentation import square, Augmenter
from multiprocessing import Queue


class DataSetLoader:

    def __init__(self,
                 data_dir="data",
                 img_size=48,
                 rotate_angle=360,
                 n_epochs=200,
                 parallel=True,
                 rng=np.random.RandomState(123)):
        self.data_dir = data_dir
        self.class_labels = {}
        self._num_label = 0
        self.img_size = img_size
        self.train_file = os.path.join("data", "tidy", "train_%d.npy" % img_size)
        self.trainlabels_file = os.path.join("data", "tidy", "train_labels_%d.npy" % img_size)
        self.test_file = os.path.join("data", "tidy", "test_%d.npy" % img_size)
        self.vanilla_file = os.path.join("data", "tidy", "vtrain_%d.npy" % img_size)
        self.vanillalabels_file = os.path.join("data", "tidy", "vtrain_labels_%d.npy" % img_size)
        self.mapfile = os.path.join("data", "tidy", "train_labmapping.pkl")
        self.trainfile = os.path.join("data", "tidy", "train.pkl")
        self.n_testimages = 130400
        # filenames in the testset order
        self.testfilenames = []
        self.rng = rng
        X, y = self.load_images()
        self.X_train, self.X_valid, self.y_train, self.y_valid = self.train_test_split(X, y)
        self.resize_f = functools.partial(resize, output_shape=(self.img_size, self.img_size))
        self.X_train_resized = np.vstack(tuple(
            [x.reshape(1, self.img_size, self.img_size)
             for x in map(self.resize_f, self.X_train)]))
        self.X_train_padded = np.vstack(tuple(map(square, self.X_train)))
        self.X_valid_padded = np.vstack(tuple(map(square, self.X_valid)))
        if parallel:
            self.queue = Queue(5)
            self.augmenter = Augmenter(self.queue,
                                       self.X_train_resized,
                                       max_items=n_epochs+1,
                                       random_seed=self.rng.randint(9999),
                                       max_angle=rotate_angle,
                                       flatten=True)
            self.augmenter.start()


    def load_images(self):
        # get cached data
        if os.path.isfile(self.trainfile) and os.path.isfile(self.mapfile):
            with open(self.mapfile, 'r') as lfile:
                self.class_labels = cPickle.load(lfile)
            with open(self.trainfile, 'r') as tfile:
                images, y = cPickle.load(tfile)
            return pd.Series(images), np.array(y, dtype='int32')
        images = []
        y = []
        for directory in sorted(glob.iglob(os.path.join(self.data_dir, "train", "*"))):
            print("processing %s" % directory)
            files = os.listdir(directory)
            n_images = len(files)
            # the last directory is a class label
            self.class_labels[self._num_label] = os.path.split(directory)[-1]
            # create labels list
            y.extend([self._num_label] * n_images)
            self._num_label += 1
            for i, image in enumerate(files):
                images.append(imread(os.path.join(directory, image), as_grey=True))
        # cache images as array for future use
        with open(self.mapfile, 'w') as lfile:
            cPickle.dump(self.class_labels, lfile)
        with open(self.trainfile, 'w') as tfile:
            cPickle.dump((images, y), tfile)
        return pd.Series(images), np.array(y, dtype='int32')

    def train_gen(self, padded=False, augment=False):
        assert len(self.X_train) == len(self.y_train)
        n_samples = len(self.X_train)
        # xs = np.zeros((n_samples, self.img_size * self.img_size), dtype='float32')
        # yield train set permutations indefinately
        while True:
            shuff_ind = self.rng.permutation(n_samples)
            if padded:
                yield self.X_train_padded[shuff_ind].astype('float32'), self.y_train[shuff_ind]
            elif augment:
                #yield self.X_train_resized[shuff_ind].astype('float32'), self.y_train[shuff_ind]
                yield self.queue.get().astype("float32"), self.y_train
            else:
                reshaped = self.X_train_resized.reshape(self.X_train_resized.shape[0], self.img_size * self.img_size)
                yield reshaped[shuff_ind].astype("float32"), self.y_train[shuff_ind]
            #transform the training set
            # xs = np.vstack(tuple(
            #      map(functools.partial(transform,
            #                            rng=self.rng,
            #                            image_size=(self.img_size, self.img_size)),
            #          self.X_train)))


    def valid_gen(self, padded=False):
        # will return same shuffled images
        while True:
            shuff_ind = self.rng.permutation(len(self.X_valid))
            if padded:
                yield self.X_valid_padded[shuff_ind].astype('float32'), self.y_valid[shuff_ind]
            else:
                xs = np.vstack(tuple([x.reshape(1,-1) for x in
                    map(functools.partial(resize, output_shape=(self.img_size, self.img_size)),
                        self.X_valid)]))
                yield xs[shuff_ind].astype('float32'), self.y_valid[shuff_ind]

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

    def train_test_split(self, X, y, test_size=0.1):
        sss = StratifiedShuffleSplit(y, n_iter=1, random_state=self.rng, test_size=test_size)
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
