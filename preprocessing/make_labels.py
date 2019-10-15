# map labels to shuffled images

import os
import yaml
import numpy as np
import scipy.io as io


def save_train_labels(tar_root_dir, train_misc_dir, train_label_dir, train_mat_dir):
    ### train LABELS ###
    train_labels = []
    train_filenames = []
    # read the labels from train.txt
    with open(os.path.join(tar_root_dir, 'train.txt'), 'r') as text_labels:
        lines = text_labels.readlines()
    for line in lines:
        filename, label = line.split()
        train_filenames.append(filename)
        train_labels.append(int(label))

    train_labels = np.asarray(train_labels)
    train_filenames = np.asarray(train_filenames)
    
    np.save(train_misc_dir, train_filenames)
    np.save(train_label_dir, train_labels)

    train_labels = {'train_labels':train_labels}
    io.savemat(train_mat_dir, train_labels)


def save_val_labels(tar_root_dir, val_misc_dir, val_label_dir, val_mat_dir):
    ### VALIDATION LABELS ###
    val_labels = []
    val_filenames = []
    with open(os.path.join(tar_root_dir, 'val.txt'), 'r') as text_labels:
        lines = text_labels.readlines()
    for line in lines:
        filename, label = line.split()
        val_filenames.append(filename)
        val_labels.append(int(label))

    val_labels = np.asarray(val_labels)
    val_filenames = np.asarray(val_filenames)

    np.save(val_misc_dir, val_filenames)
    np.save(val_label_dir, val_labels)

    val_labels = {'val_labels':val_labels}
    io.savemat(val_mat_dir, val_labels)
    
    

if __name__ == '__main__':
    with open('paths.yaml', 'r') as f:
        paths = yaml.load(f)

    tar_root_dir = paths['tar_root_dir']

    label_dir = os.path.join(tar_root_dir, 'labels')
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)
    train_label_dir = os.path.join(label_dir, 'train_labels.npy')
    val_label_dir = os.path.join(label_dir, 'val_labels.npy')  

    train_filename_dir = os.path.join(tar_root_dir, 'train_filenames.npy')
    val_filename_dir = os.path.join(tar_root_dir, 'val_filenames.npy')

    train_mat_dir = os.path.join(label_dir, 'train_labels.mat')
    val_mat_dir = os.path.join(label_dir, 'val_labels.mat')

    
    save_train_labels(tar_root_dir, train_filename_dir, train_label_dir, train_mat_dir)
    save_val_labels(tar_root_dir, val_filename_dir, val_label_dir, val_mat_dir)

