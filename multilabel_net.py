'''
Code author: Bohan Zhuang. 
Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

'''


import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
import theano.tensor as T

import numpy as np

import theano.tensor.nnet as nnet

from multilabel_layers import DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer, MultilabelLayer, ElemwiseLayer, PoolingLayer

class CNN_model(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        lib_conv = config['lib_conv']
        output_num = config['output_num']
        bias = config['bias']

        ###################### BUILD NETWORK ##########################

        x = T.ftensor4('x') 
        y = T.lmatrix('y')     # float32
        rand = T.fvector('rand')  # float32

        print '... building the model'
        self.layers = []
        params = []
        weight_types = []
        layer1_input = x       

        convpool_layer1 = ConvPoolLayer(input=layer1_input,
                                        image_shape=(3, 224, 224, batch_size), 
                                        filter_shape=(3, 3, 3, 64), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer1)
        params += convpool_layer1.params
        weight_types += convpool_layer1.weight_type

        convpool_layer2 = ConvPoolLayer(input=convpool_layer1.output,
                                        image_shape=(64, 224, 224, batch_size),
                                        filter_shape=(64, 3, 3, 64), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer2)
        params += convpool_layer2.params
        weight_types += convpool_layer2.weight_type

        convpool_layer3 = ConvPoolLayer(input=convpool_layer2.output,
                                        image_shape=(64, 112, 112, batch_size),
                                        filter_shape=(64, 3, 3, 128), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=0, 
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer3)
        params += convpool_layer3.params
        weight_types += convpool_layer3.weight_type

        convpool_layer4 = ConvPoolLayer(input=convpool_layer3.output,
                                        image_shape=(128, 112, 112, batch_size),
                                        filter_shape=(128, 3, 3, 128), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer4)
        params += convpool_layer4.params
        weight_types += convpool_layer4.weight_type

        convpool_layer5 = ConvPoolLayer(input=convpool_layer4.output,
                                        image_shape=(128, 56, 56, batch_size),
                                        filter_shape=(128, 3, 3, 256), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer5)
        params += convpool_layer5.params
        weight_types += convpool_layer5.weight_type

        convpool_layer6 = ConvPoolLayer(input=convpool_layer5.output,
                                        image_shape=(256, 56, 56, batch_size),
                                        filter_shape=(256, 3, 3, 256), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer6)
        params += convpool_layer6.params
        weight_types += convpool_layer6.weight_type

        convpool_layer7 = ConvPoolLayer(input=convpool_layer6.output,
                                        image_shape=(256, 56, 56, batch_size),
                                        filter_shape=(256, 3, 3, 256), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer7)
        params += convpool_layer7.params
        weight_types += convpool_layer7.weight_type

        convpool_layer8 = ConvPoolLayer(input=convpool_layer7.output,
                                        image_shape=(256, 28, 28, batch_size),
                                        filter_shape=(256, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer8)
        params += convpool_layer8.params
        weight_types += convpool_layer8.weight_type

        convpool_layer9 = ConvPoolLayer(input=convpool_layer8.output,
                                        image_shape=(512, 28, 28, batch_size),
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer9)
        params += convpool_layer9.params
        weight_types += convpool_layer9.weight_type

        convpool_layer10 = ConvPoolLayer(input=convpool_layer9.output,
                                        image_shape=(512, 28, 28, batch_size),
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer10)
        params += convpool_layer10.params
        weight_types += convpool_layer10.weight_type

        convpool_layer11 = ConvPoolLayer(input=convpool_layer10.output,
                                        image_shape=(512, 14, 14, batch_size),
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer11)
        params += convpool_layer11.params
        weight_types += convpool_layer11.weight_type

        convpool_layer12 = ConvPoolLayer(input=convpool_layer11.output,
                                        image_shape=(512, 14, 14, batch_size),
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=1, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer12)
        params += convpool_layer12.params
        weight_types += convpool_layer12.weight_type

        convpool_layer13 = ConvPoolLayer(input=convpool_layer12.output,
                                        image_shape=(512, 14, 14, batch_size),
                                        filter_shape=(512, 3, 3, 512), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=2, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer13)
        params += convpool_layer13.params
        weight_types += convpool_layer13.weight_type 
     

        fc_layer14_input = T.flatten(
            convpool_layer13.output.dimshuffle(3, 0, 1, 2), 2)

        fc_layer14 = FCLayer(input=fc_layer14_input, n_in=25088, n_out=4096)
        self.layers.append(fc_layer14)
        params += fc_layer14.params
        weight_types += fc_layer14.weight_type

        dropout_layer15 = DropoutLayer(fc_layer14.output, n_in=4096, n_out=4096)

        fc_layer16 = FCLayer(input=dropout_layer15.output, n_in=4096, n_out=4096)
        self.layers.append(fc_layer16)
        params += fc_layer16.params
        weight_types += fc_layer16.weight_type

        dropout_layer17 = DropoutLayer(fc_layer16.output, n_in=4096, n_out=4096)

        elem_layer18 = ElemwiseLayer(input=dropout_layer17.output, n_in=4096, n_out=output_num)
        self.layers.append(elem_layer18)
        params += elem_layer18.params
        weight_types += elem_layer18.weight_type
        
        sigmoid_output19 = T.nnet.sigmoid(elem_layer18.output)

        multilabel_layer = MultilabelLayer(sigmoid_output19, y, bias)

        
        self.cost = multilabel_layer.cost
        self.errors = multilabel_layer.error
        self.params = params
        self.weight_types = weight_types
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.predict_label = multilabel_layer.prediction
        self.output_num = output_num



def compile_models(model, config):  #introduce the Vggnet to define the gradients and updates, then construct theano functions


    model_input = model.x
    y = model.y
    weight_types = model.weight_types  #only model2 need update

    cost = model.cost
    params = model.params
    errors = model.errors
    predict_label = model.predict_label
    batch_size = model.batch_size
    output_num = model.output_num
   # test_output = model.testoutput

    mu = config['momentum']
    eta = config['weight_decay']

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []
    vels = []

    learning_rate = theano.shared(np.float32(config['learning_rate'])) #notice this form of definition
    lr = T.scalar('lr')  # symbolic learning rate, different from the tutorial 


    shared_x = theano.shared(np.zeros((3, 224, 224, batch_size), dtype=theano.config.floatX), borrow=True)

    shared_y = theano.shared(np.zeros((batch_size, output_num), dtype=int),
                             borrow=True)

    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]


    # construct the updates list by looping over all the parameters
    if config['use_momentum']:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in \
                zip(params, grads, vels, weight_types):

            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = lr
            elif weight_type == 'W_ful':        
                real_grad = grad_i + eta * param_i
                real_lr = 10. * lr
            elif weight_type == 'b_ful':
                read_grad = grad_i
                read_lr = 10. * lr
            else:
                raise TypeError("Weight Type Error")


            if config['use_nesterov_momentum']:
                vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
            else:
                vel_i_next = mu * vel_i - real_lr * real_grad  #correspond to the update role of the paper

            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i + vel_i_next))  # update each model parameter param_i and vel_i

    else:
        for param_i, grad_i, weight_type in zip(params, grads, weight_types):
            if weight_type == 'W':
                updates.append((param_i,
                                param_i - lr * grad_i - eta * lr * param_i))
            elif weight_type == 'b':
                updates.append((param_i, param_i - 2 * lr * grad_i))
            else:
                raise TypeError("Weight Type Error")


   
    train_model = theano.function([], cost, updates=updates,
                                  givens=[(model_input, shared_x), (y, shared_y), (lr, learning_rate)])

    validate_model = theano.function([], [cost, errors],
                                     givens=[(model_input, shared_x), (y, shared_y)])

    predict_model = theano.function([], [predict_label], givens=[(model_input, shared_x)])

    train_error = theano.function(
        [], errors, givens=[(model_input, shared_x), (y, shared_y)])

    return (train_model, validate_model, predict_model, train_error,
            learning_rate, shared_x, shared_y, vels)
