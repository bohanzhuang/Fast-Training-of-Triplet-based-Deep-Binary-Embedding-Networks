
# Preprocessing: From JPEG to HKL

import os
import glob
import sys
import yaml
import scipy.misc
import numpy as np
import hickle as hkl


def get_img(img_name, img_size=227, batch_size=64):

    target_shape = (img_size, img_size, 3)
    img = scipy.misc.imread(img_name)  # x*x*3
    assert img.dtype == 'uint8', img_name
    # assert False

    if len(img.shape) == 2:  #gray-scale image
        img = scipy.misc.imresize(img, (img_size, img_size))
        img = np.asarray([img, img, img])  
    else:
        if img.shape[2] > 3: #special image
            img = img[:, :, :3]
        img = scipy.misc.imresize(img, target_shape)
        img = np.rollaxis(img, 2)   
    if img.shape[0] != 3:
        print img_name
    return img  


def save_batches(file_list, tar_dir, img_size=227, batch_size=64,
                 flag_avg=False, num_sub_batch=1):   #If flag_avg ture, then calculate the image_mean of the training data.  Save the training and validation data patch
 

    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    img_batch = np.zeros((3, img_size, img_size, batch_size), np.uint8)

    if flag_avg:
        img_sum = np.zeros((3, img_size, img_size))

    batch_count = 0
    count = 0
    for file_name in file_list:
        img_batch[:, :, :, count % batch_size] = \
            get_img(file_name, img_size=img_size, batch_size=batch_size) 

        count += 1
        if count % batch_size == 0:
            batch_count += 1

            if flag_avg:
                img_sum += img_batch.mean(axis=3)

            if num_sub_batch == 1:
                save_name = '%04d' % (batch_count - 1) + '.hkl'
                hkl.dump(img_batch, os.path.join(tar_dir, save_name), mode='w') 

    return img_sum / batch_count if flag_avg else None  # this simple presentation



if __name__ == '__main__':
    with open('paths.yaml', 'r') as f:
        paths = yaml.load(f)

    tar_root_dir = paths['tar_root_dir']
    train_filenames = np.load(tar_root_dir + '/' + 'train_img_names.npy')
    val_filenames = np.load(tar_root_dir + '/' + 'val_img_names.npy')

    tar_train_dir = tar_root_dir + 'train_hkl'
    tar_val_dir = tar_root_dir + 'test_hkl'


    img_size = 227
    batch_size = 50
    num_sub_batch = 1


   # generate training data
    img_mean = save_batches(train_filenames, tar_train_dir,
                           img_size=img_size, batch_size=batch_size,
                           flag_avg=True, num_sub_batch=num_sub_batch)

    np.save(os.path.join(tar_root_dir, 'img_mean.npy'), img_mean)

    # validation data
    save_batches(val_filenames, tar_val_dir,
                 img_size=img_size, batch_size=batch_size,
                 num_sub_batch=num_sub_batch)


