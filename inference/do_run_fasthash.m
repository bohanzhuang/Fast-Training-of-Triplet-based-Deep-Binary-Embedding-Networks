% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com


function do_run_fasthash(train_info, infer_info, update_bit, group_idx, sign)


  
if update_bit == 1
       
    old_hamming_distances = zeros(train_info.triplet_samples_num, 2);
    new_hamming_distances = old_hamming_distances; 
    
end
    
if sign == true
    
   if group_idx == 1 
   pre_group_hamming_distances = zeros(train_info.triplet_samples_num, 2);
   save(['./temp/pre_group_hamming_distances_' num2str(group_idx) '.mat'], 'pre_group_hamming_distances');
        
   else
   
   if ~exist(['../temp/hash_step2_code_' num2str(group_idx-1) '.mat'], 'file')
   unix('python ../train.py');    
   end
       
   pre_group_code = load(['../temp/hash_step2_code_' num2str(group_idx-1) '.mat']);
   load(['./temp/pre_group_hamming_distances_' num2str(group_idx-1) '.mat']);
  
   pre_group_code = int8(pre_group_code.hash_step2_code);  
   pre_group_code(find(pre_group_code == 0)) = -1;
    
    group_hamm_dist_pairs = calc_hamm_dist_group(pre_group_code, train_info.triplet_samples_idxes);
    
    new_hamming_distances = group_hamm_dist_pairs;
    pre_group_hamming_distances = new_hamming_distances;
    
    save(['./temp/pre_group_hamming_distances_' num2str(group_idx) '.mat'], 'pre_group_hamming_distances');
    
   end
else
    
   temp_distance = load(['./temp/old_hamming_distances_' num2str(update_bit-1) '.mat']);
   old_hamming_distances = temp_distance.old_hamming_distances;
    
   pre_bi_code = load(['./temp/hash_step1_code_' num2str(update_bit - 1) '.mat']);
   pre_bi_code = pre_bi_code.hash_step1_code;
   pre_bi_code(find(pre_bi_code == 0)) = -1;
   
   assert(~isempty(pre_bi_code));
   current_hamming_distances = calculate_hamming_distances(train_info.triplet_samples_idxes, pre_bi_code);
   new_hamming_distances = old_hamming_distances + current_hamming_distances; 
      
    
end

if ~exist(['./temp/hash_step1_code_' num2str(update_bit) '.mat'])
    
    % construct the loss dictionary
    t1 = tic;
    loss_dictionary = construct_loss_dictionary(update_bit);
    
      
    % construct the pairwise coefficients dictionary
    pairwise_coefficients_dictionary = construct_coefficients_dictionary(loss_dictionary);
          
    
    % construct potential function coefficients matrix (pairwise matrix and unary matrix) 
    % unary_coefficients_matrix: N*3   pairwise_coefficients_matrix: N * N
    % unary_coefficients: two parts!!!
    [pairwise_coefficients_matrix, pairwise_coefficients_vector] = construct_pairwise_coefficients_matrix(new_hamming_distances, pairwise_coefficients_dictionary, infer_info, train_info);
    infer_info.pairwise_weights_matrix = pairwise_coefficients_matrix;
    infer_info.pairwise_weights_vector = pairwise_coefficients_vector;
    
    
    % construct block graphcut
    infer_info.infer_block_info = gen_infer_block(pairwise_coefficients_matrix, train_info);
   

    % use block graphcut to calculate hash_code
    infer_info.init_bi_code = ones(train_info.samples_num, 1);
   % init_bi_code = round(unifrnd(0,1,1,train_info.samples_num))';
   % init_bi_code(init_bi_code==0)=-1;
   % infer_info.init_bi_code = init_bi_code;
    
    infer_info.e_num = train_info.samples_num;
    
    infer_result = do_infer_block(train_info, infer_info);
    
    fprintf('\n------------- do_infer_block finished \n')
    
   % train the decision tree classifier
 
   hash_step1_code = infer_result.infer_bi_code;
   idxes = find(hash_step1_code == -1);
   hash_step1_code(idxes) = 0;
    
   save(['./temp/hash_step1_code_' num2str(update_bit)], 'hash_step1_code', '-v7.3');
    
   old_hamming_distances = new_hamming_distances;
   save(['./temp/old_hamming_distances_' num2str(update_bit)], 'old_hamming_distances', '-v7.3'); 
   
   inference_time = toc(t1);
   save('inference_time', 'inference_time');
   
   
end
end

    

 function group_hamm_dist_pairs = calc_hamm_dist_group(pre_group_code, relation_map)

    group_hamm_dist_pairs = zeros(size(relation_map, 1), 2);

    for i = 1:size(pre_group_code, 2)
        
        bi_code = pre_group_code(:,i);

        left_bi_code=bi_code(relation_map(:,1),:);
        middle_bi_code = bi_code(relation_map(:,2),:);
        right_bi_code=bi_code(relation_map(:,3),:);   
        
        group_hamm_dist_pairs(:,1) = group_hamm_dist_pairs(:,1) + sum(left_bi_code~=middle_bi_code, 2);
        group_hamm_dist_pairs(:,2) = group_hamm_dist_pairs(:,2) + sum(left_bi_code~=right_bi_code, 2);
        
        
    end



end
   
   
   
   
   



