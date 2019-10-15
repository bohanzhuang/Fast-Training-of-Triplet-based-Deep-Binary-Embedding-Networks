

function [hash_learner hash_learners_model cache_info hlearner_bi_code]=gen_cdboost(...
    learner_param, bi_train_data, hash_learners_model, cache_info)

    if ~isfield(cache_info, 'is_init') || ~cache_info.is_init
        cache_info.is_init=true;
        hash_learners_model.apply_hash_learner_fn=@apply_cdboost;
    end
        
        
    train_result=cdboost_train(learner_param, bi_train_data);
    model=train_result.model;

    hash_learner=[];
    hash_learner{1}=model;
	hlearner_bi_code=apply_cdboost(bi_train_data.feat_data, hash_learner, hash_learners_model);
	
  
        
end
