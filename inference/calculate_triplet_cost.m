% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function loss = calculate_triplet_cost(infer_bi_code, train_info)

triplet_samples_idxes = train_info.triplet_samples_idxes;

triplet_code_map = infer_bi_code(triplet_samples_idxes);

distance_1 = triplet_code_map(:,1) ~= triplet_code_map(:,3);

distance_2 = triplet_code_map(:,1) ~= triplet_code_map(:,2);

distance = distance_1 - distance_2;

loss = sum(max(0, 0.5 - distance));




end
