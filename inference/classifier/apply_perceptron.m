

function bi_code=apply_perceptron(feat_data, hash_learners, hash_learners_model)
    
    learner_idxes=cell2mat(hash_learners);
    model_params=hash_learners_model.model_params(learner_idxes,:);

    assert(~issparse(model_params));
    % assert(~issparse(feat_data));


    assert(isa(feat_data, 'double') || isa(feat_data, 'single'));
    if issparse(feat_data)
        model_params=sparse(model_params);
    end
        
    
    if size(model_params, 2)>size(feat_data,2)
        assert(size(model_params, 2)==size(feat_data,2)+1);
        tmp_v=feat_data*model_params(:,1:end-1)';
        tmp_v=tmp_v+repmat(model_params(:, end)', size(tmp_v, 1), 1);
        feat_code=tmp_v>0;
        
    else
        feat_code=feat_data*model_params'>0;
    end
    
    
    bi_code=gen_bi_code_from_logical(feat_code);
      
end
