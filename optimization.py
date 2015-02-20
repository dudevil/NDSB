__author__ = 'dudevil'

"""
This file contains code for generating SGD updates with regular and Nesterov momentums
"""

import theano
import numpy as np


def gen_updates_regular_momentum(loss, all_parameters, learning_rate, momentum, weight_decay=0.0):
    """
    Stohastic gradient descent (SGD) with regular momentum

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
        mparam_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX),
                                 broadcastable=param_i.broadcastable)
        if param_i in all_bias_parameters:
            full_grad = grad_i
        else:
            full_grad = grad_i + weight_decay * param_i
        v = momentum * mparam_i - learning_rate * full_grad # new momemtum
        w = param_i + momentum * v - learning_rate * full_grad # new parameter values
        updates.append((mparam_i, v))
        updates.append((param_i, w))
    return updates