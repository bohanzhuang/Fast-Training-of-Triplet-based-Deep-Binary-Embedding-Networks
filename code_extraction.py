import sys
import time
import os

import yaml
import numpy as np
import pycuda.driver as drv
import subprocess
import scipy.io as sio
import h5py
import glob



sys.path.append('./lib')  # add necessary path
from tools import (save_weights, load_weights, load_weights_finetune, load_feature_extraction_weights, initialize_weights, save_momentums, load_momentums)
from train_funcs import (unpack_configs, proc_configs, get_prediction_labels)


def code_extraction(config):

    # UNPACK CONFIGS
    (train_filenames, val_filenames, img_mean) = unpack_configs(config)   

    import theano.sandbox.cuda
    theano.sandbox.cuda.use(config['gpu'])

    import theano
    theano.config.on_unused_input = 'warn'
    import theano.tensor as T

    from multilabel_layers import DropoutLayer
    from multilabel_net import CNN_model, compile_models

    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils

    model = CNN_model(config)
    batch_size = model.batch_size
    layers = model.layers


    n_train_batches = len(train_filenames)
    n_val_batches = len(val_filenames)


        ## COMPILE FUNCTIONS ##
    (predict_model, shared_x) = compile_models(model, config)
    
    load_weights_epoch = 8

    train_predicted_code = None
    val_predicted_code = None

    load_weights_dir = config['weights_dir']

    load_weights(layers, load_weights_dir, load_weights_epoch) 

    code_save_dir = config['code_save_dir']

    DropoutLayer.SetDropoutOff()   


    for minibatch_index in range(n_train_batches):           

        label = get_prediction_labels(predict_model, shared_x, train_filenames, minibatch_index, img_mean)
        
        if train_predicted_code is None:
        	train_predicted_code = label[0]
        else:
        	train_predicted_code = np.vstack((train_predicted_code, label[0]))

    database_code = {'database_code': train_predicted_code}
    sio.savemat(code_save_dir + 'database_code.mat', database_code) 


    for minibatch_index in range(n_val_batches):           

        label = get_prediction_labels(predict_model, shared_x, val_filenames, minibatch_index, img_mean)
        
        if val_predicted_code is None:
        	val_predicted_code = label[0]
        else:
        	val_predicted_code = np.vstack((val_predicted_code, label[0]))


    test_code = {'test_code': val_predicted_code}
    sio.savemat(code_save_dir + 'test_code.mat', test_code)  


    print('code extraction complete.')



if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        
    config = proc_configs(config)

    code_extraction(config)
