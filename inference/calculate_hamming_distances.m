% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function current_hamming_distance = calculate_hamming_distances(triplet_samples_idxes, pre_hash_code)


hash_code_map = pre_hash_code(triplet_samples_idxes);

distance_1 = sum(hash_code_map(:,2) ~= hash_code_map(:,1), 2);
distance_2 = sum(hash_code_map(:,3) ~= hash_code_map(:,1), 2);

current_hamming_distance = cat(2, distance_1, distance_2);
