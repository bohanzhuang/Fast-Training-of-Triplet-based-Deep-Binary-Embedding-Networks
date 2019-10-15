% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function [train_info, infer_info] = gen_train_info(train_data)



% generate train_info

train_info.samples_num = size(train_data.label_data, 1);
train_info.infer_block_type = 'graphcut';
train_info.infer_iter_num= 1;


infer_info = [];

disp('generate affinity information...');

triplet_samples_idxes = gen_affinity_labels(train_data);

train_info.triplet_samples_num = size(triplet_samples_idxes, 1);


train_info.triplet_samples_idxes = triplet_samples_idxes;


[pairwise_relation_map_symmetric, pairwise_relation_map] = gen_pairwise_map(train_info, triplet_samples_idxes);

infer_info.pairwise_relation_map_symmetric = pairwise_relation_map_symmetric;
infer_info.pairwise_relation_map = pairwise_relation_map;


end
