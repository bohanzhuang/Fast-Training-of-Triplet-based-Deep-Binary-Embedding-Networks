'''
Load data in parallel with train.py
Code author: Bohan Zhuang. 
Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

'''

import time
import math

import numpy as np
#import zmq
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import hickle as hkl


def get_params_crop_and_mirror(param_rand, data_shape, cropsize):

    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = round(param_rand[0] * center_margin * 2)  # round: to the closest integer
    crop_ys = round(param_rand[1] * center_margin * 2)
    if False:
        # this is true then exactly replicate Ryan's code, in the batch case
        crop_xs = math.floor(param_rand[0] * center_margin * 2) # floor: the largest interger less/equal to x
        crop_ys = math.floor(param_rand[1] * center_margin * 2)

    flag_mirror = bool(round(param_rand[2]))

    return crop_xs, crop_ys, flag_mirror


def center_crop(data, param_rand, data_shape, cropsize=224):
    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = round(param_rand[0] * center_margin * 2)  # round: to the closest integer
    crop_ys = round(param_rand[1] * center_margin * 2)
    data = data[:, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :]

#    return np.ascontiguousarray(data, dtype='float32') 
    return np.asarray(data, dtype='float32')



def crop_and_mirror(data, param_rand, flag_batch=True, cropsize=227):
    '''
    when param_rand == (0.5, 0.5, 0), it means no randomness
    '''
    # print param_rand

    # if param_rand == (0.5, 0.5, 0), means no randomness and do validation
    # in training stage, use get_rand3d() to generate random variables
    if param_rand[0] == 0.5 and param_rand[1] == 0.5 and param_rand[2] == 0:
        flag_batch = True

    if flag_batch:
        # mirror and crop the whole batch
        crop_xs, crop_ys, flag_mirror = \
            get_params_crop_and_mirror(param_rand, data.shape, cropsize)

        # random mirror
        if flag_mirror:
            data = data[:, :, ::-1, :]

        # random crop
        data = data[:, crop_xs:crop_xs + cropsize,
                    crop_ys:crop_ys + cropsize, :]

    else:
        # mirror and crop each batch individually
        # to ensure consistency, use the param_rand[1] as seed
        np.random.seed(int(10000 * param_rand[1]))

        data_out = np.zeros((data.shape[0], cropsize, cropsize,
                                data.shape[3])).astype('float32') #notice this form of definition

        for ind in range(data.shape[3]):
            # generate random numbers
            tmp_rand = np.float32(np.random.rand(3))
            tmp_rand[2] = round(tmp_rand[2])

            # get mirror/crop parameters
            crop_xs, crop_ys, flag_mirror = \
                get_params_crop_and_mirror(tmp_rand, data.shape, cropsize)

            # do image crop/mirror
            img = data[:, :, :, ind]
            if flag_mirror:
                img = img[:, :, ::-1]
            img = img[:, crop_xs:crop_xs + cropsize,
                      crop_ys:crop_ys + cropsize]
            data_out[:, :, :, ind] = img

        data = data_out

    return np.ascontiguousarray(data, dtype='float32')  #return a contiguous array in c01b order

