__author__ = 'dudevil'

import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams
from read_data import DataSetLoader


# Rectified linear unit activation
def relu(x):
    return T.maximum(x, 0)

# max activation function, applied to HiddenLayer results in a MaxoutLayer
#
# def maxout(x):
#     # take a maximum for each sample in a minibatch (that's what 1 stands for)
#     theano.printing.Print(x.get_value().shape)
#     return T.max(x, 1)


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
                W_values = numpy.asarray(
                    rng.normal(
                        loc=0.0,
                        scale=numpy.sqrt(2./n_out),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
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
            col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
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
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
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
    BC01 version of CrossChannelNormalization
    Parameters
    ----------
    alpha : WRITEME
    k : WRITEME
    beta : WRITEME
    n : WRITEME
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
            scale += self.alpha * sq[:,i:i+ch,:,:]

        scale = scale ** self.beta

        return bc01 / scale

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape,
                 poolsize=(2, 2),
                 stride=(1,1),
                 normalize=False,
                 activation=relu):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
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
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
        #            numpy.prod(poolsize))
        #
        # initialize weights with random weights
        # for relu units use initialization proposed here:
        # http://arxiv-web3.library.cornell.edu/pdf/1502.01852v1.pdf
        if activation == relu:
            W_var = numpy.sqrt(2. / fan_in)
            W_bound = numpy.asarray(
                rng.normal(loc=0.0, scale=W_var, size=filter_shape),
                dtype=theano.config.floatX)
        else:
            # weights are initialized from a gaussian distribution centered at 0 with varience 0.01
            W_bound = numpy.asarray(
                rng.normal(loc=0.0, scale=0.01, size=filter_shape),
                dtype=theano.config.floatX)
        self.W = theano.shared(W_bound, borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # parametric relu activation
        if activation == 'prelu':
            self.a = theano.shared(numpy.array((0.25,), dtype=theano.config.floatX), borrow=False, name='a', broadcastable=(True,))
            #print(self.a)
        else:
            self.a = theano.shared(numpy.array(0.0, dtype=theano.config.floatX), borrow=False, name='a')
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            subsample=stride

        )
        if normalize:
            conv_out = CrossChannelNormalizationBC01()(conv_out)

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

    def __init__(self, input, a_init=0.25):
        self.input = input
        self.a = theano.shared(numpy.cast[theano.config.floatX](a_init), name='a')
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


class DropOutLayer(object):
    """
    Currently not used, needs refactoring
    TO-DO: refactor
    """

    def __init__(self, rng, input, input_shape, active, rate=0.5):
        rstream = RandomStreams(seed=rng.randint(9999))
        if active.get_value(borrow=True):
            # enable dropout for training
            mask = T.cast(rstream.binomial(n=1, p=rate, size=input_shape),
                          theano.config.floatX)
            theano.printing.Print("Dropout mask mean: " % mask.mean())
            self.output = input * mask
        else:
            theano.printing.Print("No dropout")
            # disable dropout for prediction
            self.output = input

# dropout currenly in use
# this should be refactored into a class like above ^^^
# for more information on dropout see: http://jmlr.org/papers/v15/srivastava14a.html
def dropout(rng, input, input_shape, active, rate=0.5):
        rstream = RandomStreams(seed=rng.randint(9999))
        mask = T.cast(rstream.binomial(n=1, p=rate, size=input_shape),
                          theano.config.floatX)

        out = T.switch(active, mask * input / rate, input)
        return out



def gen_updates_regular_momentum(loss, all_parameters, learning_rate, momentum, weight_decay=0.0):
    """
    Stohastic gradient descent (SGD) with regular momentum

    Nesterov momentum (below) works better so never really tried this out
    """
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value()*0.)
        v = momentum * mparam_i - weight_decay * learning_rate * param_i - learning_rate * grad_i
        updates.append((mparam_i, v))
        updates.append((param_i, param_i + v))
    return updates

def gen_updates_nesterov_momentum_no_bias_decay(loss,
                                                all_parameters,
                                                all_bias_parameters,
                                                learning_rate,
                                                momentum,
                                                weight_decay=0.0005):
    """
    Nesterov momentum, but excluding the biases from the weight decay.
    If biases are included learning the network seems impossible.
    For more info on Nesterov momentum see: www.cs.toronto.edu/~fritz/absps/momentum.pdf

    Implementation taken from: https://github.com/benanne/kaggle-galaxies
    """
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value()*0.)
        if param_i in all_bias_parameters:
            full_grad = grad_i
        else:
            full_grad = grad_i + weight_decay * param_i
        v = momentum * mparam_i - learning_rate * full_grad # new momemtum
        w = param_i + momentum * v - learning_rate * full_grad # new parameter values
        updates.append((mparam_i, v))
        updates.append((param_i, w))
    return updates

learning_rate_schedule = {
    0: 0.01,
    100: 0.005,
    150: 0.001,
    160: 0.0009,
    170: 0.0008,
    180: 0.0007,
    190: 0.0005,
    300: 0.0008,
    400: 0.0005,
    450: 0.0001
}

momentum_schedule = {
    0: 0.9,
    102: 0.95,
    150: 0.99,
    302: 0.99,
    400: 0.995
}

def evaluate_lenet5(n_epochs=200, nkerns=[32, 64, 128, 128], batch_size=200):
    """ Demonstrates lenet on MNIST dataset
    Build train and evaluate model
    """

    rng = numpy.random.RandomState(121)
    print "Preparing datasets ..."
    # load data and split train/test set stratified by classes
    dsl = DataSetLoader(rng=rng, img_size=49, n_epochs=n_epochs, parallel=True, pad=True)
    train_gen = dsl.train_gen(augment=True)
    valid_gen = dsl.valid_gen()
    train_x, train_y = train_gen.next()
    valid_x, valid_y = valid_gen.next()
    # test_x = dsl.load_test()
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.shape[0] / batch_size
    n_val_batches = valid_x.shape[0] / batch_size
    # n_test_batches = test_x.shape[0] / batch_size

    train_x = theano.shared(train_x, borrow=True)
    valid_x = theano.shared(valid_x, borrow=True)
    # won't fit into my gpu memory

    train_y = T.cast(theano.shared(train_y, borrow=True), dtype='int32')
    valid_y = T.cast(theano.shared(valid_y, borrow=True), dtype='int32')

    # allocate learning rate and momentum shared variables
    learning_rate = theano.shared(numpy.array(learning_rate_schedule[0], dtype=theano.config.floatX))
    momentum = theano.shared(numpy.array(momentum_schedule[0], dtype=theano.config.floatX))

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    dropout_active = T.bscalar('dropout_active')  # a flag to enable and disable dropout



    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print 'Building the model ...'

    # Reshape matrix of rasterized images of shape (batch_size, 48 * 48)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 49, 49))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to ((49-8)+1 , (49-8)+1) = (42, 42)
    # maxpooling reduces this further to (42/2, 42/2) = (21, 21)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 21, 21)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 49, 49),
        filter_shape=(nkerns[0], 1, 8, 8),
        poolsize=(2, 2),
        normalize=True,
        activation=relu
    )

    #prlayer0 = ParametrizedReLuLayer(layer0.output)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (21-4+1, 21-4+1) = (18, 18)
    # maxpooling reduces this further to (18/2, 18/2) = (9, 9)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 14, 14)
    #layer1_input = dropout(rng, layer0.output, (batch_size, nkerns[0], 22, 22), dropout_active, rate=0.1)
    #layer1_input = prlayer.output
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 21, 21),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2),
        normalize=True,
        activation=relu
    )

    #prlayer1 = ParametrizedReLuLayer(layer1.output)
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (9-4+1, 9-4+1) = (6, 6)
    #layer2_input = dropout(rng, layer1.output, (batch_size, nkerns[1], 9, 9), dropout_active, rate=0.1)
    layer2_input = layer1.output
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer2_input,
        image_shape=(batch_size, nkerns[1], 9, 9),
        filter_shape=(nkerns[2], nkerns[1], 4, 4),
        poolsize=(),
        activation=relu
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (6-3+1, 6-3+1) = (4, 4)
    # maxpooling reduces this further to (4/2, 4/2) = (2, 2)
    #prlayer2 = ParametrizedReLuLayer(layer2.output)
    layer7 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 6, 6),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(2, 2),
        activation=relu
    )

    #prlayer3 = ParametrizedReLuLayer(layer7.output)
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 5 * 5),

    layer3_input = dropout(rng, layer7.output.flatten(2), (batch_size, nkerns[3] * 2 * 2), dropout_active)
    # construct a fully-connected relu layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[3] * 2 * 2,
        n_out=2048,
        activation=relu,
        max_col_norm=2.
    )
    # prlayer4 = ParametrizedReLuLayer(layer3.output)
    # Maxout layer reduces output dimension to (batch_size, input_dim / pool_size)
    # in this case: (batch_size, 512/2) = (batch_size, 256)
    # maxlayer1 = MaxOutLayer(
    #      input=layer3.output,
    #      input_shape=(batch_size, 2560),
    #      pool_size=2
    # )
    # add dropout at 0.5 rate
    layer4_input = dropout(rng, layer3.output, (batch_size, 2048), dropout_active)

    # Maxout layer reduces output dimension to (batch_size, input_dim / pool_size)
    # in this case: (batch_size, 512/2) = (batch_size, 256)
    # one more fully-connected relu layer
    layer4 = HiddenLayer(
        rng,
        input=layer4_input,
        n_in=2048,
        n_out=2048,
        activation=relu,
        max_col_norm=2.
    )
    # Maxout layer reduces output dimension to (batch_size, input_dim / pool_size)
    # in this case: (batch_size, 512/2) = (batch_size, 256)
    # maxlayer2 = MaxOutLayer(
    #     input=layer4.output,
    #     input_shape=(batch_size, 2560),
    #     pool_size=2
    # )
    #prlayer5 = ParametrizedReLuLayer(layer4.output)
    # add dropout at 0.5 rate
    layer6_input = dropout(rng, layer4.output, (batch_size, 2048), dropout_active)
    # classify the values of the fully-connected sigmoidal layer
    layer6 = LogisticRegression(input=layer6_input, n_in=2048, n_out=121)

    # the cost we minimize during training is the NLL of the model
    cost = layer6.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    # this is basically a fraction of incorrectly classified images
    test_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: valid_x[index * batch_size: (index + 1) * batch_size],
            y: valid_y[index * batch_size: (index + 1) * batch_size],
            dropout_active: theano.shared(numpy.array(0, dtype='int8'), borrow=False)
        }
    )

    # create a function to compute the multi-class logarithmic loss wich is the evaluation metric
    # for this competition (it's the same as negative loglikelyhood)
    test_logloss = theano.function(
        [index],
        layer6.negative_log_likelihood(y),
        givens={
            x: valid_x[index * batch_size: (index + 1) * batch_size],
            y: valid_y[index * batch_size: (index + 1) * batch_size],
            dropout_active: theano.shared(numpy.array(0, dtype='int8'), borrow=False)
        }
    )


    # create a list of all model parameters to be fit by gradient descent
    params = layer6.params + layer4.params + layer2.params + layer1.params + layer0.params +\
             layer3.params + layer7.params


    # a list of bias parameters: these will be excluded from the Nesterov momentum updates
    bias_params = layer6.bias_params + layer4.bias_params + layer2.bias_params + layer1.bias_params \
                  + layer0.bias_params + layer3.bias_params + layer7.bias_params
    # create a list of gradients for all model parameters, this would be a plain SGD
    #grads = T.grad(cost, params)
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    # updates = [
    #     (param_i, param_i - learning_rate * grad_i)
    #     for param_i, grad_i in zip(params, grads)
    # ]

    # we generate the updates with Nesterov momentum
    updates = gen_updates_nesterov_momentum_no_bias_decay(cost, params,
                                                          bias_params,
                                                          learning_rate=learning_rate,
                                                          momentum=momentum)

    # apply max-norm regularization
    # layer3.censor_updates(updates)
    # layer4.censor_updates(updates)

    # create a function to train the neural network
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size],
            dropout_active: theano.shared(numpy.array(1, dtype='int8'), borrow=False),
        }
    )

    # save train and validation errors for future analysis
    valid_err = []
    test_err = []
    n_iter = []

    ###############
    # TRAIN MODEL #
    ###############
    print 'Training ...'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_test_logloss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    print("| Epoch | Train err | Validation err | Validation misclass | Ratio |")
    print("|------------------------------------------------------------------|")
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            # if iter % 100 == 0:
            #     print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            # update learning rate and momentum according to schedule
            if epoch in learning_rate_schedule:
                 learning_rate.set_value(numpy.array(learning_rate_schedule[epoch], dtype=theano.config.floatX))
            if epoch in momentum_schedule:
                 momentum.set_value(numpy.array(momentum_schedule[epoch], dtype=theano.config.floatX))

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                test_losses = [test_model(i) for i
                                     in xrange(n_val_batches)]
                test_loglosses = [test_logloss(i) for i
                                     in xrange(n_val_batches)]
                this_test_loss = numpy.mean(test_losses)
                this_test_logloss = numpy.mean(test_loglosses)
                print("|%6d | %9.6f | %14.6f | %17.2f %% | %1.3f |" %
                      (epoch, cost_ij, this_test_logloss, this_test_loss * 100., this_test_logloss / cost_ij))
                # print('epoch %i, minibatch %i/%i, validation error %.2f %% loglikelihood %f train error %f' %
                #       (epoch, minibatch_index + 1, n_train_batches,
                #        this_test_loss * 100., this_test_logloss, cost_ij))
                # print('Max weight in dense layers 1: %f 2: %f' %
                #       (numpy.max(numpy.sqrt(numpy.sum(numpy.square(layer3.W.get_value(borrow=True)), axis=0))),
                #        numpy.max(numpy.sqrt(numpy.sum(numpy.square(layer4.W.get_value(borrow=True)), axis=0)))))

                # save for later analysis
                valid_err.append(cost_ij)
                test_err.append(this_test_logloss)
                n_iter.append(epoch)
                # if we got the best validation score until now
                if this_test_logloss < best_test_logloss:

                    #improve patience if loss improvement is good enough
                    if this_test_logloss < best_test_logloss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_test_logloss = this_test_logloss
                    best_test_loss = this_test_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break
        #print("[%f] epoch done" % time.clock())
        # get training data for next epoch
        nx, ny = train_gen.next()
        #print("[%f] train epoch generated" % time.clock())
        # load the data into the GPU
        train_x.set_value(nx, borrow=True)
        train_y.set_value(ny, borrow=True)
        #print("[%f] train epoch loaded" % time.clock())

    end_time = time.clock()
    print('Optimization complete.')
    print('Best logloss score of %f %% obtained at iteration %i (epoch %i), '
          'with misclassification rate %f %%' %
          (best_test_logloss, best_iter + 1, best_iter / n_train_batches + 1, best_test_loss * 100.))
    print >> sys.stderr, ('The training lasted %.2fm' % ((end_time - start_time) / 60.))
    ################
    #  Predicting  #
    ################
    # res = []
    # print(test_x.shape)
    #
    # n_test_batches = test_x.shape[0] / batch_size
    # test_x = theano.shared(test_x, borrow=True)
    #
    # predict = theano.function(
    #     [index],
    #     layer6.predict_proba(),
    #     givens={
    #         x: test_x[index * batch_size: (index + 1) * batch_size],
    #         dropout_active: theano.shared(numpy.array(0, dtype='int8'), borrow=False)
    #         },
    # )
    # for minibatch_index in xrange(n_test_batches):
    #         tmp = predict(minibatch_index)
    #         res.append(tmp)
    # result = numpy.vstack(tuple(res))
    # print(result.shape)
    # dsl.save_submission(result, '2')

    # save train and validation errors
    results = numpy.array([n_iter, test_err, valid_err], dtype=numpy.float)
    numpy.save("data/tidy/49_4convlayers_dense2048_norm01.npy", results)

if __name__ == '__main__':
    evaluate_lenet5()
