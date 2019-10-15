

function [hash_learner hash_learners_model cache_info hlearner_bi_code]=gen_svm_linear(...
    learner_param, bi_train_data, hash_learners_model, cache_info)

    if ~isfield(cache_info, 'is_init') || ~cache_info.is_init
        
        % feat_data_sparse=sparse(bi_train_data.feat_data);
        % cache_info.feat_data_sparse=feat_data_sparse;
        
        hash_learners_model.model_params=zeros(learner_param.bit_num, size(bi_train_data.feat_data, 2)+1);
        hash_learners_model.apply_hash_learner_fn=@apply_perceptron;
        
        cache_info.is_init=true;
        
    end
        
    
    label_data=bi_train_data.label_data;
    feat_data=bi_train_data.feat_data;

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
    
    hlearner_bi_code=apply_perceptron(bi_train_data.feat_data, hash_learner, hash_learners_model);
        
end
