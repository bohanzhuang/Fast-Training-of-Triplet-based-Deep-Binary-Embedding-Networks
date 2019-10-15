% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com


function triplet_samples_idxes = gen_train_samples(triplet_counter, affinity_labels, train_info)


[similar_idxes, h_idx] = find(affinity_labels'==1);

[disimilar_idxes, ~] = find(affinity_labels'==-1);

triplet_samples_idxes = zeros(triplet_counter, 3);


 for i = 1: train_info.samples_num
     
     idxes = find(h_idx==i);
    
     for j = idxes(1):idxes(end)
         
        triplet_samples_idxes(j,1) = h_idx(j);
        triplet_samples_idxes(j,2) = similar_idxes(j);
        triplet_samples_idxes(j,3) = disimilar_idxes(j);
         
     end
       
     
 end



end
