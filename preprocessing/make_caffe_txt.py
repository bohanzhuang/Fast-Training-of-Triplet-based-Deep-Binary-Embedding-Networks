import os
import yaml
import scipy.io
import numpy as np
import sys



with open('paths.yaml', 'r') as f:
	paths = yaml.load(f)


train_img_dir  = paths['train_img_dir']
val_img_dir = paths['val_img_dir']
misc_dir = paths['tar_root_dir']



traintxt_filename = misc_dir + 'train.txt'
valtxt_filename = misc_dir + 'val.txt'


sorted_train_dirs = sorted([name for name in os.listdir(train_img_dir)
                            if os.path.isdir(os.path.join(train_img_dir, name))])
train_sorted_id_list = range(len(sorted_train_dirs))
train_dict_wnid_to_sorted_id = {sorted_train_dirs[ind]: ind   
                          for ind in train_sorted_id_list}


sorted_val_dirs = sorted([name for name in os.listdir(val_img_dir)
                            if os.path.isdir(os.path.join(val_img_dir, name))])
val_sorted_id_list = range(len(sorted_val_dirs))
val_dict_wnid_to_sorted_id = {sorted_val_dirs[ind]: ind   
                          for ind in val_sorted_id_list}



train_filenames = []
val_filenames =[]
train_img_names = []
val_img_names = []
temp_train_filenames = []
temp_val_filenames = []
sub_train_names = []
sub_val_names = []



for folder in sorted_train_dirs:
    for name in os.listdir(os.path.join(train_img_dir, folder)):
        temp_train_filenames += sorted(
        [train_img_dir + folder + '/' + name + ' ' + str(train_dict_wnid_to_sorted_id[folder]) + '\n'])
        sub_train_names += sorted([train_img_dir + folder + '/' + name])

train_idx = range(len(sub_train_names))
seed = 1
np.random.seed(seed)
np.random.shuffle(train_idx)
for idx in train_idx:
    train_filenames.append(temp_train_filenames[idx])
    train_img_names.append(sub_train_names[idx])



for folder in sorted_val_dirs:
    for name in os.listdir(os.path.join(val_img_dir, folder)):
        temp_val_filenames += sorted(
        [val_img_dir + folder + '/' + name + ' ' + str(val_dict_wnid_to_sorted_id[folder]) + '\n'])
        sub_val_names += sorted([val_img_dir + folder + '/' + name])

val_idx = range(len(sub_val_names))
seed = 1
np.random.seed(seed)
np.random.shuffle(val_idx)
for idx in val_idx:
    val_filenames.append(temp_val_filenames[idx])
    val_img_names.append(sub_val_names[idx])


np.save('./preprocessed_data/train_img_names.npy', train_img_names)
np.save('./preprocessed_data/val_img_names.npy', val_img_names)

    

with open(traintxt_filename, 'w') as f:
    f.writelines(train_filenames)

with open(valtxt_filename, 'w') as f:
    f.writelines(val_filenames)
