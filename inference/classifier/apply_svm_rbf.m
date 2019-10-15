


function bi_code=apply_svm_rbf(feat_data, hash_learners, hash_learners_model)


    learner_idxes=cell2mat(hash_learners);
    model_infos=hash_learners_model.model_infos(learner_idxes);
    

    bit_num=size(hash_learners, 1);
    bi_code=zeros(size(feat_data, 1), bit_num, 'int8');
    
    feat_data_kernel=apply_kernel_feat(hash_learners_model, feat_data);
    
    
    for b_idx=1:length(model_infos)
        
        
        one_model_info=model_infos{b_idx};
        sv_indices=one_model_info.sv_indices;
        sv_coef=one_model_info.sv_coef;
        b=one_model_info.b;
        
        svs=feat_data_kernel(:, sv_indices);
        feat_code = svs*sv_coef + b  > 0;
    
        one_bi_code=gen_bi_code_from_logical(feat_code);
        bi_code(:, b_idx)=one_bi_code;
    end
    
      
end






function feat_data_kernel=apply_kernel_feat(hash_learners_model, feat_data)
   
   
    % should not use sparse value, it would be very slow.
    % assert(~issparse(feat_data));
    support_vectors=hash_learners_model.support_vectors;

    
    feat_data_kernel = sqdist(feat_data',support_vectors');
    feat_data_kernel = exp(-feat_data_kernel/(2*hash_learners_model.sigma));
    

end



