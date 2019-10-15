% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function [pairwise_relation_map_symmetric, pairwise_relation_map] = gen_pairwise_map(train_info, triplet_samples_idxes)

column_1 = [];
column_2 = [];
column_3 = [];
column_4 = [];

for i = 1:train_info.triplet_samples_num
    
   triplet_idx = triplet_samples_idxes(i,:);
   triplet_idx_1 = [triplet_idx(1); triplet_idx(1); triplet_idx(2); triplet_idx(2); triplet_idx(3); triplet_idx(3)];
   triplet_idx_2 = [triplet_idx(2); triplet_idx(3); triplet_idx(1); triplet_idx(3); triplet_idx(1); triplet_idx(2)];
   triplet_idx_3 = [triplet_idx(1); triplet_idx(1); triplet_idx(2)];
   triplet_idx_4 = [triplet_idx(2); triplet_idx(3); triplet_idx(3)];
   
   column_1 = cat(1, column_1, triplet_idx_1);
   column_2 = cat(1, column_2, triplet_idx_2);
   
   column_3 = cat(1, column_3, triplet_idx_3);
   column_4 = cat(1, column_4, triplet_idx_4);
    
end

pairwise_relation_map_symmetric = [column_1, column_2];
pairwise_relation_map = [column_3, column_4];


end

