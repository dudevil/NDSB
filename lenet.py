__author__ = 'dudevil'

import os
import sys
import time

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from read_data import DataSetLoader

def relu(x):
    return x * (x > 0)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=relu):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

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
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
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
            W_values = numpy.asarray(
                rng.normal(
                    loc=0.0,
                    scale=0.001,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            # initialize biases to 0.01
            b_values = numpy.ones((n_out,), dtype=theano.config.floatX) / 100
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class DropOutLayer(object):

    def __init__(self, input, input_shape, rng, rate=0.5):
        rstream = RandomStreams(seed=rng.randint(9999))
        mask = rstream.binomial(n=1, p=rate, size=input_shape)
        self.output = input * T.cast(mask, theano.config.floatX)

class MaxPoolLayer(object):

    def __init__(self, input, pool_size=(2, 2)):
        # downsample each feature map individually, using maxpooling
        self.output = downsample.max_pool_2d(
            input=input,
            ds=pool_size,
            ignore_border=True
        )



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
        # start-snippet-1
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

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
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
        # end-snippet-2

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
        return self.p_y_given_x


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, activation=relu):
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

        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # initialize weights with gaussian with mean 0 and deviation 0.01
        self.W = theano.shared(
            numpy.asarray(
                rng.normal(loc=0, scale=0.01, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map, initialize to 0.1
        b_values = numpy.ones((filter_shape[0],), dtype=theano.config.floatX) / 10
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


# using the alternative formulation of nesterov momentum described at https://github.com/lisa-lab/pylearn2/pull/136
# such that the gradient can be evaluated at the current parameters.
def gen_updates_nesterov_momentum(loss, all_parameters, learning_rate, momentum, weight_decay):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value()*0.)
        full_grad = grad_i + weight_decay * param_i
        v = momentum * mparam_i - learning_rate * full_grad # new momemtum
        w = param_i + momentum * v - learning_rate * full_grad # new parameter values
        updates.append((mparam_i, v))
        updates.append((param_i, w))
    return updates

def gen_updates_regular_momentum(loss, all_parameters, learning_rate, momentum, weight_decay):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value()*0.)
        v = momentum * mparam_i - weight_decay * learning_rate * param_i - learning_rate * grad_i
        updates.append((mparam_i, v))
        updates.append((param_i, param_i + v))
    return updates

def evaluate_lenet5(learning_rate=0.1, n_epochs=500,
                    nkerns=[64, 96, 96], batch_size=200):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(010215)

    # load data and split train/test set stratified by classes
    dsl = DataSetLoader()
    train_x, valid_x, train_y, valid_y = dsl.train_test_split(random_state=rng)
    test_x = dsl.load_test()

    train_xx = train_x # tmp
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.shape[0] / batch_size
    n_val_batches = valid_x.shape[0] / batch_size
    n_test_batches = test_x.shape[0] / batch_size

    train_x = theano.shared(train_x, borrow=True)
    valid_x = theano.shared(valid_x, borrow=True)
    # won't fit into my gpu memory
    #test_x = theano.shared(test_x, borrow=True)
    train_y = T.cast(theano.shared(train_y, borrow=True), dtype='int32')
    valid_y = T.cast(theano.shared(valid_y, borrow=True), dtype='int32')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 48 * 48)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 48, 48))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (48-5+1 , 48-5+1) = (44, 44)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 48, 48),
        filter_shape=(nkerns[0], 1, 5, 5)
    )

    # pooling reduces space from (44/3, 44/3) to (14,14)
    layer1 = MaxPoolLayer(
        input=layer0.output,
        pool_size=(3, 3)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (14-3+1, 14-3+1) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 2, 2)
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[0], 14, 14),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
    )

    # (12-3+1, 12-3+1) = (10,10)
    layer3 = LeNetConvPoolLayer(
          rng,
          input=layer2.output,
          image_shape=(batch_size, nkerns[1], 12, 12),
          filter_shape=(nkerns[2], nkerns[1], 3, 3),
    )

    # (10/3,10/3) = (3, 3)
    layer4 = MaxPoolLayer(
        input=layer3.output,
        pool_size=(3, 3)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 4 * 4),
    # or (500, 20 * 4 * 4) = (500, 1620) with the default values.
    layer5_input = layer4.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer5 = HiddenLayer(
        rng,
        input=layer5_input,
        n_in=nkerns[2] * 3 * 3,
        n_out=512,
        activation=relu
    )

    layer6 = DropOutLayer(layer5.output, (512,), rng)

    layer7 = HiddenLayer(
        rng,
        input=layer6.output,
        n_in=512,
        n_out=512,
        activation=relu
    )

    layer8 = DropOutLayer(layer7.output, (512,), rng)

    # classify the values of the fully-connected sigmoidal layer
    layer9 = LogisticRegression(input=layer8.output, n_in=512, n_out=121)

    # the cost we minimize during training is the NLL of the model
    cost = layer9.negative_log_likelihood(y) + 0.0001 * (T.sum(T.sqr(layer5.W)) + T.sum(T.sqr(layer7.W))) / batch_size

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer9.errors(y),
        givens={
            x: valid_x[index * batch_size: (index + 1) * batch_size],
            y: valid_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_logloss = theano.function(
        [index],
        layer9.negative_log_likelihood(y),
        givens={
            x: valid_x[index * batch_size: (index + 1) * batch_size],
            y: valid_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    # create a list of all model parameters to be fit by gradient descent
    params = layer0.params + layer2.params + layer3.params + \
             layer5.params + layer9.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    # replace Add Nesterov momentum
    #updates = gen_updates_regular_momentum(cost, params, learning_rate, 0.9, 0.0)

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size]
        },
    )

    train_err = []
    test_err = []
    n_iter = []

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
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

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                test_losses = [test_model(i) for i
                                     in xrange(n_val_batches)]
                test_loglosses = [test_logloss(i) for i
                                     in xrange(n_val_batches)]
                this_test_loss = numpy.mean(test_losses)
                this_test_logloss = numpy.mean(test_loglosses)
                print('epoch %i, minibatch %i/%i, validation error %.2f loglikelihood %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_test_loss * 100., this_test_logloss))

                # save for later analysis
                train_err.append(cost_ij)
                test_err.append(this_test_logloss)
                n_iter.append(iter + 1)

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

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with loglikelihood %f %%' %
          (best_test_loss * 100., best_iter + 1, best_test_logloss))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ### Predicting
    ###
    # res = []
    # print(test_x.shape)
    # parts = numpy.array_split(test_x, 4)
    # test_x_part = theano.shared(parts[0], borrow=True)
    # predict = theano.function(
    #     [index],
    #     layer3.predict_proba(),
    #     givens={
    #         x: test_x_part[index * batch_size: (index + 1) * batch_size],
    #         },
    # )
    # for part in parts:
    #     test_x_part.set_value(part, borrow=True)
    #     n_part_batches = part.shape[0] / batch_size
    #     for minibatch_index in xrange(n_part_batches):
    #         tmp = predict(minibatch_index)
    #         res.append(tmp)
    # result = numpy.vstack(tuple(res))
    # print(result.shape)
    # dsl.save_submission(result, '1')


    #print("Saving: %s %s %s" % (n_iter, test_err, train_err))
    results = numpy.array([n_iter, test_err, train_err], dtype=numpy.float)
    numpy.save("data/tidy/relu_errors.npy", results)

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)