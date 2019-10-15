

function bi_code=apply_svm_rbf_kernel_feat(feat_data, hash_learners, hash_learners_model)


    feat_data=apply_kernel_feat(hash_learners_model, feat_data);
    
    bi_code=apply_perceptron(feat_data, hash_learners, hash_learners_model);
      
end






function feat_data=apply_kernel_feat(hash_learners_model, feat_data)
   
   
    % should not use sparse value, it would be very slow.
    % assert(~issparse(feat_data));
    
    support_vectors=hash_learners_model.support_vectors;

    
    feat_data_kernel = sqdist(feat_data',support_vectors');
    feat_data_kernel = exp(-feat_data_kernel/(2*hash_learners_model.sigma));
    feat_data_kernel = bsxfun(@minus, feat_data_kernel, hash_learners_model.feat_data_mean);

    feat_data=feat_data_kernel;


end



