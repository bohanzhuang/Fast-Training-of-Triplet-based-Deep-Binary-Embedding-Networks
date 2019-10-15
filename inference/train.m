% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function train()

clc; clear;


addpath(genpath([pwd '/']));

bits_per_group = 8;
bits_num = 64;
total_group = bits_num / bits_per_group;

group_idx = 1:total_group
    
update_bit = (group_idx-1) * bits_per_group;
sign = true;
save('./temp/group_idx', 'group_idx');
  
for j = 1: bits_per_group
     
  update_bit = update_bit + 1;

 if update_bit == 1
     
 ds_train_file = './preprocessed_data/train_labels.mat';
 disp('load dataset...');
 temp_train=load(ds_train_file);
     
 train_datapath = './preprocessed_data/train_hkl/';
 train_frameDir = dir(train_datapath);
 train_e_num = (length(train_frameDir)-2) * 50 ;  
 ds.train_labels = temp_train.train_labels(:,1:train_e_num)';
 train_data.label_data=single(ds.train_labels);  
 [train_info, infer_info] = gen_train_info(train_data);
 train_info.bit_num = bits_num;
% 
% 
% %save triplet samples information
 save('./temp/train_info', 'train_info', '-v7.3');
 save('./temp/infer_info', 'infer_info', '-v7.3');

    
else
    
load ./temp/train_info.mat;
load ./temp/infer_info.mat;    
       
end

do_run_fasthash(train_info, infer_info, update_bit, group_idx, sign);
sign = false;

  end
end


% train the last group
t2 = tic;

group_idx = group_idx + 1;
save('./temp/group_idx', 'group_idx');
unix('python ../train.py'); 


end




