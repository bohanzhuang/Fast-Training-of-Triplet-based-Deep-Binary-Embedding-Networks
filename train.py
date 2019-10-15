'''
Code author: Bohan Zhuang. 
Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

'''

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



sys.path.append('../lib')  # add necessary path
from tools import (save_weights, load_weights, load_weights_finetune, load_feature_extraction_weights, initialize_weights, save_momentums, load_momentums)
from train_funcs import (unpack_configs, adjust_learning_rate,
                         get_val_error_loss, get_rand3d, train_model_wrap,
                         proc_configs, get_prediction_labels, extract_features)


def train_net(config):

    # UNPACK CONFIGS
    (train_filenames, img_mean) = unpack_configs(config)   


    import theano.sandbox.cuda
    theano.sandbox.cuda.use(config['gpu'])

    import theano
    theano.config.on_unused_input = 'warn'
    import theano.tensor as T

    from multilabel_layers import DropoutLayer
    from multilabel_net import CNN_model, compile_models

    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils



   # load hash_step1_bits
    group_idx = sio.loadmat('./inference/temp/group_idx.mat')
    group_idx = group_idx['group_idx']
    group_idx = group_idx[0][0]
    code_per_group = 8

    bits_idxes = range((group_idx-1) * code_per_group)

    config['output_num'] = len(bits_idxes)

    model = CNN_model(config)

    batch_size = model.batch_size
    layers = model.layers
    weight_types = model.weight_types
    params = model.params



    val_filenames = train_filenames[:20]

    n_train_batches = len(train_filenames)
    minibatch_range = range(n_train_batches)



        ## COMPILE FUNCTIONS ##
    (train_model, validate_model, predict_model, train_error, learning_rate,
        shared_x, shared_y, vels) = compile_models(model, config)


    train_labels = None
        
    for idx in bits_idxes:

        hash_step1_code = h5py.File('./inference/temp/hash_step1_code_' + str(idx + 1) + '.mat')
        temp = np.transpose(np.asarray(hash_step1_code['hash_step1_code'])).astype('int64')

        if train_labels is None:
        	train_labels = temp
        else:
        	train_labels = np.hstack([train_labels, temp])

    train_labels[train_labels==-1] = 0

    val_labels = train_labels[:20*batch_size]      

            ######################### TRAIN MODEL ################################

    print '... training'


#    initialize_weights(layers, weight_types)
#    learning_rate.set_value(config['learning_rate'])

#    vels = [theano.shared(param_i.get_value() * 0.)
#            for param_i in params]

        
    # Start Training Loop
    epoch = 0
    step_idx = 0
    val_record = []
    predicted_labels = None
    while epoch < config['n_epochs']:
        epoch = epoch + 1

        if config['shuffle']:
            np.random.shuffle(minibatch_range)

        if config['finetune'] and epoch == 1 and not config['resume_train']:
        	load_weights_finetune(layers, config['finetune_weights_dir'])

        count = 0
        for minibatch_index in minibatch_range:

            num_iter = (epoch - 1) * n_train_batches + count
            count = count + 1
            if count == 1:
                s = time.time()
            if count == 20:
                e = time.time()
                print "time per 20 iter:", (e - s)

            cost_ij = train_model_wrap(train_model, shared_x,
                                       shared_y, minibatch_index,
                                       minibatch_range, batch_size,
                                       train_labels, train_filenames, img_mean)


            if num_iter % config['print_freq'] == 0:
                print 'training @ iter = ', num_iter
                print 'training cost:', cost_ij
                if config['print_train_error']:
                    print 'training error rate:', train_error()


        ############### Test on Validation Set ##################

        DropoutLayer.SetDropoutOff()

        this_validation_error, this_validation_loss = get_val_error_loss(shared_x, shared_y, img_mean, val_filenames, val_labels,
        	                                                             batch_size, validate_model)


        print('epoch %i: validation loss %f ' %
              (epoch, this_validation_loss))
        print('epoch %i: validation error %f %%' %
              (epoch, this_validation_error * 100.))
        val_record.append([this_validation_error, this_validation_loss])

        savepath = config['weights_dir'] + 'classifier_' + str(group_idx - 1) + '/'
        if not os.path.exists(savepath):
            	os.mkdir(savepath)
                
        np.save(savepath + 'val_record.npy', val_record)

        DropoutLayer.SetDropoutOn()

        ############################################

        # Adapt Learning Rate
        step_idx = adjust_learning_rate(config, epoch, step_idx,
                                        val_record, learning_rate)

        # Save weights for each iteration
        if epoch % 5 == 0:
  
            save_weights(layers, savepath, epoch)
            np.save(savepath + 'lr_' + str(epoch) + '.npy',
                   learning_rate.get_value())
            save_momentums(vels, savepath, epoch)


    DropoutLayer.SetDropoutOff() 
                # generate the labels
    for minibatch_index in range(n_train_batches):           

        label = get_prediction_labels(predict_model, shared_x, minibatch_index, train_filenames, img_mean)
        if predicted_labels is None:
        	predicted_labels = label[0]
        else:
        	predicted_labels = np.vstack((predicted_labels, label[0]))


    hash_step2_code = {'hash_step2_code': predicted_labels}
    sio.savemat('./temp/hash_step2_code_' + str(group_idx - 1) + '.mat', hash_step2_code) 

    DropoutLayer.SetDropoutOn()
  

    print('Optimization complete.')


if __name__ == '__main__':

    with open('../config.yaml', 'r') as f:
        config = yaml.load(f)
        
    config = proc_configs(config)

    train_net(config)

