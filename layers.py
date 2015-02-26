__author__ = 'dudevil'

import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import numpy as np


# Rectified linear unit activation
def relu(x):
    return T.maximum(x, 0)


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=relu, max_col_norm=0.0):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The default nonlinearity used here is relu

        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            if activation == relu:
                # for relu units use initialization proposed here:
                # http://arxiv-web3.library.cornell.edu/pdf/1502.01852v1.pdf
                W_values = np.asarray(
                    rng.normal(
                        loc=0.0,
                        scale=np.sqrt(2./n_out),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX)
            else:
                W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # set up max-norm regularization
        self.max_col_norm = max_col_norm
        # parameters of the model
        self.params = [self.W, self.b]
        self.bias_params = [self.b]

    def censor_updates(self, updates):
        """
        When applied to the updates dictionary this function causes the weights to stay
        in a sphere with radius self.max_col_norm
        :param updates:
        :return:
        """
        W = self.W
        if W in updates and self.max_col_norm:
            updated_W = updates[W]
            col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0, keepdims=True))
            desired_norms = T.clip(col_norms, 0, self.max_col_norm)
            constrained_W = updated_W * (desired_norms / (1e-7 + col_norms))
            updates[W] = constrained_W


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
        self.bias_params = [self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def predict_proba(self):
        """
        Predict conditional probability of class membership given observations
        :return:
        """
        return self.p_y_given_x


class CrossChannelNormalizationBC01(object):
    """
    This corresponds to local response normalization mention in the Krizhevsky paper:
    www.cs.toronto.edu/~fritz/absps/imagenet.pdf Section 3.3

    This code is taken from the pylearn2 library:
    https://github.com/lisa-lab/pylearn2/blob/14b2f8bebce7cc938cfa93e640008128e05945c1/pylearn2/expr/normalize.py
    """

    def __init__(self, alpha = 1e-4, k=2, beta=0.75, n=5):
        self.__dict__.update(locals())
        del self.self

        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n for now")

    def __call__(self, bc01):
        """
        .. todo::
            WRITEME
        """
        half = self.n // 2

        sq = T.sqr(bc01)

        b, ch, r, c = bc01.shape

        extra_channels = T.alloc(0., b, ch + 2*half, r, c)

        sq = T.set_subtensor(extra_channels[:,half:half+ch,:,:], sq)

        scale = self.k

        for i in xrange(self.n):
            scale += self.alpha * sq[:, i:i+ch, :, :]

        scale = scale ** self.beta

        return bc01 / scale


class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape,
                 poolsize=(2, 2),
                 stride=(1, 1),
                 normalize=False,
                 activation=relu):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
        #            np.prod(poolsize))
        #
        # initialize weights with random weights
        # for relu units use initialization proposed here:
        # http://arxiv-web3.library.cornell.edu/pdf/1502.01852v1.pdf
        if activation == relu:
            W_var = np.sqrt(2. / fan_in)
            W_bound = np.asarray(
                rng.normal(loc=0.0, scale=W_var, size=filter_shape),
                dtype=theano.config.floatX)
        else:
            # weights are initialized from a gaussian distribution centered at 0 with varience 0.01
            W_bound = np.asarray(
                rng.normal(loc=0.0, scale=0.01, size=filter_shape),
                dtype=theano.config.floatX)
        self.W = theano.shared(W_bound, borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        self.ccn = CrossChannelNormalizationBC01()

        # parametric relu activation
        if activation == 'prelu':
            self.a = theano.shared(np.array((0.25,), dtype=theano.config.floatX), borrow=False, name='a', broadcastable=(True,))
            #print(self.a)
        else:
            self.a = theano.shared(np.array(0.0, dtype=theano.config.floatX), borrow=False, name='a')
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            subsample=stride

        )
        if normalize:
            conv_out = self.ccn(conv_out)

        if poolsize:
            # downsample each feature map individually, using maxpooling
            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True
            )
        else:
            # no downsampling
            pooled_out = conv_out

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        if activation:
            self.output = activation(lin_output)
        else:
            self.output = lin_output

        # store parameters of this layer
        self.params = [self.W, self.b]
        self.bias_params = [self.b]


class ParametrizedReLuLayer(object):
    """
    An attempt to implement parametrized ReLUs from this paper: http://arxiv.org/abs/1502.01852

    This implementation is broken.
    """

    def __init__(self, input, a_init=0.25):
        self.input = input
        self.a = theano.shared(np.cast[theano.config.floatX](a_init), name='a')
        self.output = T.maximum(0, input) + self.a * T.minimum(0, input)
        self.params = [self.a]
        self.bias_params = []


class MaxOutLayer(object):
    """
    This class selects max feature value from the input vector with stride pool_size
    When placed on top of a dense layer with linear activation (e.g. no activation)
    it's equivalent to the Maxout architecture described here:
    http://www-etud.iro.umontreal.ca/~goodfeli/maxout.html
    """

    def __init__(self, input, input_shape, pool_size=2):
        self.input = input
        # assume 2D matrix as input (like after a hidden layer) [batch_size, n_features]
        b_size, n_in = input_shape
        # number of inputs should be divisible by pool_size
        assert n_in % pool_size == 0
        # reshape the input into a 3D tensor, the last dimension (2) will be maxed-out
        input_reshaped = input.reshape((b_size, n_in / pool_size, pool_size))
        self.output = T.max(input_reshaped, 2)
        self.output_shape = (b_size, n_in // pool_size)


class DropOutLayer(object):
    """
    Currently not used, needs refactoring
    """

    def __init__(self, rng, input, input_shape, active, rate=0.5):
        rstream = RandomStreams(seed=rng.randint(9999))
        mask = T.cast(rstream.binomial(n=1, p=rate, size=input_shape),
                          theano.config.floatX)

        self.output = T.switch(active, mask * input / rate, input)
        self.output_shape = input_shape


class SliceLayer(object):

    def __init__(self, input, input_shape, out_imgshape=(48, 48)):
        batch_size, channels, x, y = input_shape
        nx, ny = out_imgshape
        # upper left
        part0 = input[:, :, :nx, :ny]
        # upper right
        part1 = input[:, :, -nx:, :ny]
        # lower left
        part2 = input[:, :, :nx, -ny:]
        # lower right
        part3 = input[:, :, -nx:, -ny:]
        # center
        x_offset = (x - nx) // 2  # int division
        y_offset = (y - ny) // 2  # int division
        part4 = input[:, :, x_offset:-x_offset, y_offset:-y_offset]
        # concatenate
        self.output = T.concatenate([part0, part1, part2, part3, part4], axis=0)
        # save number of sliced parts for later merge (see next class)
        self.n_parts = 5


class MergeLayer(object):

    def __init__(self, input, input_shape, n_parts):
        batch_size, channels, kernels, x, y = input_shape
        assert batch_size % n_parts == 0  # this should be divisible
        # reshape into 2 d
        self.output_shape = (batch_size // n_parts, channels * kernels * x * y * n_parts)
        self.output = input.reshape(self.output_shape)
