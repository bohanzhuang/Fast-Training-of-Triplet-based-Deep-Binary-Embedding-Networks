% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com


function [ pairwise_coefficients_matrix, pairwise_coefficients_vector]  = construct_pairwise_coefficients_matrix(new_hamming_distances,pairwise_coefficients_dictionary, infer_info, train_info)

% get the unary and pairwise coefficients
% get pairwise relation map (not symmetric)

values = [];

for i = 1:train_info.triplet_samples_num

   
   % get the pairwise weights
   
   pairwise_coefficients = pairwise_coefficients_dictionary{new_hamming_distances(i,1)+1, new_hamming_distances(i,2)+1};
  
   values = cat(1, values, pairwise_coefficients);
   
     
end

% make the pairwise_coefficients_matrix symmetric
pairwise_coefficients_matrix = sparse(infer_info.pairwise_relation_map_symmetric(:,1), infer_info.pairwise_relation_map_symmetric(:,2), values, train_info.samples_num, train_info.samples_num);

pairwise_coefficients_matrix = (pairwise_coefficients_matrix + pairwise_coefficients_matrix')/2;

%bias = train_info.lambda * ones(size(pairwise_coefficients_matrix));

%pairwise_coefficients_matrix = pairwise_coefficients_matrix + bias; 


% generate pairwise coefficients vector

pairwise_idxes = sub2ind(size(pairwise_coefficients_matrix), infer_info.pairwise_relation_map(:,1), infer_info.pairwise_relation_map(:,2));

pairwise_coefficients_vector = pairwise_coefficients_matrix(pairwise_idxes);




end
