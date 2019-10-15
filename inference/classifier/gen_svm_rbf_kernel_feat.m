

function [hash_learner hash_learners_model cache_info hlearner_bi_code]=gen_svm_rbf_kernel_feat(...
    learner_param, bi_train_data, hash_learners_model, cache_info)

    if ~isfield(cache_info, 'is_init') || ~cache_info.is_init
        
        feat_data=bi_train_data.feat_data;
        [kernel_feat_data hash_learners_model]=gen_kernel_feat(learner_param, hash_learners_model, feat_data);
        
        cache_info.kernel_feat_data=kernel_feat_data;
        
        % feat_data_sparse=sparse(kernel_feat_data);
        % cache_info.feat_data_sparse=feat_data_sparse;
        
        hash_learners_model.model_params=zeros(learner_param.bit_num, size(kernel_feat_data, 2)+1);
        hash_learners_model.apply_hash_learner_fn=@apply_svm_rbf_kernel_feat;
        
        cache_info.is_init=true;
        
    end
    
       
    
    
    label_data=bi_train_data.label_data;
    feat_data=cache_info.kernel_feat_data;

    e_num=length(label_data);    
    tradeoff_param=learner_param.tradeoff_param/e_num;
    
    params=sprintf('-q -B 1 -s 2 -c %.6f ', tradeoff_param);
    
    svm_model = my_liblinear_train(label_data, feat_data, params);

    w=svm_model.w;
    if svm_model.Label(1)<0
        w=-w;
    end
    
    learner_idx=bi_train_data.hash_learner_idx;
    hash_learners_model.model_params(learner_idx,:)=w;
    hash_learner{1}=learner_idx;
        
    
    hlearner_bi_code=apply_perceptron(cache_info.kernel_feat_data, hash_learner, hash_learners_model);
    
    
end




function [feat_data_kernel hash_learners_model]=gen_kernel_feat(learner_param, hash_learners_model, feat_data)

      
    sigma=learner_param.sigma;
    support_vectors=learner_param.support_vectors;     
    assert(~isempty(support_vectors));
    
    feat_data_kernel = sqdist(feat_data',support_vectors');
    feat_data_kernel = exp(-feat_data_kernel/(2*sigma));
    feat_data_mean = mean(feat_data_kernel);
    feat_data_kernel = bsxfun(@minus, feat_data_kernel, feat_data_mean);
    
    hash_learners_model.support_vectors=support_vectors;
    hash_learners_model.sigma=sigma;
    hash_learners_model.feat_data_mean=feat_data_mean;
                
end




