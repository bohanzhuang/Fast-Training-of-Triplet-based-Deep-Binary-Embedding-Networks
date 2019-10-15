'''
Code author: Bohan Zhuang. 
Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

'''

import glob
import time
import os

import numpy as np
import scipy.io as sio
import hickle as hkl
from proc_load import center_crop



def proc_configs(config):
    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])
        print "Creat folder: " + config['weights_dir']

    return config


def unpack_configs(config, ext_data='.hkl', ext_label='.npy'):


    # Load Training/Validation Filenames and Labels

    train_folder = config['train_folder']
    train_filenames = sorted(glob.glob(train_folder + '/*' + ext_data))

    img_mean = np.load(config['mean_file'])
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32') 

    return (train_filenames, img_mean)





def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt Learning Rate
    if config['lr_policy'] == 'step':
        if epoch == config['lr_step'][step_idx]:
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))   #learning_rate is the shared variable
            step_idx += 1
            if step_idx >= len(config['lr_step']):
                step_idx = 0  # prevent index out of range error
            print 'Learning rate changed to:', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3] - val_record[-1] <
                            config['lr_adapt_threshold']):
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            print 'Learning rate changed to::', learning_rate.get_value()

    return step_idx


def get_val_error_loss(shared_x, shared_y, img_mean,
                       val_filenames, val_labels,
                       batch_size, validate_model):


    validation_losses = []
    validation_errors = []

    n_val_batches = len(val_filenames)


    for val_index in range(n_val_batches):

        batch_img = hkl.load(str(val_filenames[val_index])) - img_mean
#        param_rand = np.float32([0.5, 0.5, 0])
#        batch_img = center_crop(batch_img, param_rand, batch_img.shape)
        shared_x.set_value(batch_img)

        shared_y.set_value(val_labels[val_index * batch_size:
                                      (val_index + 1) * batch_size])
        loss, error = validate_model()

        # print loss, error
        validation_losses.append(loss)
        validation_errors.append(error)

    this_validation_loss = np.mean(validation_losses)
    this_validation_error = np.mean(validation_errors)

    return this_validation_error, this_validation_loss


def get_rand3d():
    tmp_rand = np.float32(np.random.rand(3))
    tmp_rand[2] = round(tmp_rand[2])
    return tmp_rand



def extract_features(feature_extraction_model, shared_x, img_mean,
                     minibatch_index, batch_size, train_filenames):

    batch_img = hkl.load(str(train_filenames[minibatch_index])) - img_mean
    batch_img = np.asarray(batch_img, dtype='float32')
    shared_x.set_value(batch_img)
    features = feature_extraction_model()
    return features

    

def train_model_wrap(train_model, shared_x, shared_y, minibatch_index, minibatch_range, batch_size,
                     train_labels, train_filenames, img_mean):   #give value to shared_x, shared_y, rand_arr


    #set value to shared_x, shared_y, rand_arr shared symbolic variables
    batch_img = hkl.load(str(train_filenames[minibatch_index])) - img_mean
#    param_rand = np.float32([0.5, 0.5, 0])
#    batch_img = center_crop(batch_img, param_rand, batch_img.shape)
    shared_x.set_value(batch_img)
    
    #load the training_label
    batch_label = train_labels[minibatch_index * batch_size:
                              (minibatch_index + 1) * batch_size]

    shared_y.set_value(batch_label)
 
    cost_ij = train_model()

    return cost_ij


def small_debug(debug_model, shared_x, minibatch_range, minibatch_index, train_filenames, img_mean):
    batch_img = hkl.load(str(train_filenames[minibatch_index])) - img_mean
    batch_img = np.asarray(batch_img, dtype='float32')
    shared_x.set_value(batch_img)
    debug_output = debug_model()
    return debug_output


def get_prediction_labels(predict_model, shared_x, minibatch_index, train_filenames, img_mean):

    batch_img = hkl.load(str(train_filenames[minibatch_index])) - img_mean
#    param_rand = np.float32([0.5, 0.5, 0])
#    batch_img = center_crop(batch_img, param_rand, batch_img.shape)
    shared_x.set_value(batch_img)
    predict_label = predict_model()
    return predict_label




