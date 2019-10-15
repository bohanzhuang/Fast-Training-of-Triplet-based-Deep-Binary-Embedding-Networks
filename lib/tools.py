'''
Code author: Bohan Zhuang. 
Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

'''


import os
import numpy as np
import math

def save_weights(layers, weights_dir, epoch):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.save_weight(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.save_weight(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.save_weight(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.save_weight(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.save_weight(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.save_weight(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))


def load_weights(layers, weights_dir, epoch):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.load_weight(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.load_weight(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.load_weight(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.load_weight(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.load_weight(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.load_weight(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))



def initialize_weights(layers, weight_types):

    count = 0
    for idx in range(len(weight_types)/2):

        if  weight_types[count] =='W':
            layers[idx].W.initialize_weight(0, 0.01, 'W_' + str(idx), layers[idx].W.val.get_value().shape)
            layers[idx].b.initialize_weight(0, 0.01, 'b_' + str(idx), layers[idx].b.val.get_value().shape)

        if  weight_types[count] =='W_ful':
            layers[idx].W.initialize_weight(0, 0.005, 'W_' + str(idx), layers[idx].W.val.get_value().shape)
            layers[idx].b.initialize_weight(0.1, 0, 'b_' + str(idx), layers[idx].b.val.get_value().shape)

        if  weight_types[count] =='W_softmax':
            layers[idx].W.initialize_weight(0, 0.01, 'W_' + str(idx), layers[idx].W.val.get_value().shape)
            layers[idx].b.initialize_weight(0, 0, 'b_' + str(idx), layers[idx].b.val.get_value().shape)
            
        count = count + 2         




def load_weights_finetune(layers, weights_dir):
    for idx in range(len(layers)-3):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.load_weight(
                weights_dir, 'W' + '_' + str(idx))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.load_weight(
                weights_dir, 'W0' + '_' + str(idx))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.load_weight(
                weights_dir, 'W1' + '_' + str(idx))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.load_weight(
                weights_dir, 'b' + '_' + str(idx))


def load_feature_extraction_weights(layers, weights_dir):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.load_weight(
                weights_dir, 'W' + '_' + str(idx))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.load_weight(
                weights_dir, 'b' + '_' + str(idx))
        if hasattr(layers[idx], 'W_ful'):
            layers[idx].b.load_weight(
                weights_dir, 'W_ful' + '_' + str(idx))         
        


def save_momentums(vels, weights_dir, epoch):
    for ind in range(len(vels)):
        np.save(os.path.join(weights_dir, 'mom_' + str(ind) + '_' + str(epoch)),
                vels[ind].get_value())


def load_momentums(vels, weights_dir, epoch):
    for ind in range(len(vels)):
        vels[ind].set_value(np.load(os.path.join(
            weights_dir, 'mom_' + str(ind) + '_' + str(epoch) + '.npy')))
