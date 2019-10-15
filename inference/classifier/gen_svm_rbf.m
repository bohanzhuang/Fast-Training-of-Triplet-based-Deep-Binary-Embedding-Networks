

function [hash_learner hash_learners_model cache_info hlearner_bi_code]=gen_svm_rbf(...
    learner_param, bi_train_data, hash_learners_model, cache_info)

    if ~isfield(cache_info, 'is_init') || ~cache_info.is_init

    	assert(isa(bi_train_data.feat_data, 'double'));

        feat_data_sparse=sparse(double(bi_train_data.feat_data));
        cache_info.feat_data_sparse=feat_data_sparse;
        
        hash_learners_model.support_vectors=bi_train_data.feat_data;
        hash_learners_model.sv_sel=false(size(feat_data_sparse, 1), 1);
        
        hash_learners_model.sigma=learner_param.sigma;
        hash_learners_model.model_infos=cell(learner_param.bit_num, 1);
        hash_learners_model.apply_hash_learner_fn=@apply_svm_rbf;
        hash_learners_model.post_process_fn=@do_post_process;
        
        
        
        cache_info.is_init=true;
        
    end
        
    
    label_data=bi_train_data.label_data;
    feat_data=cache_info.feat_data_sparse;


    assert(~issparse(label_data));
    assert(isa(label_data, 'double'));


    e_num=length(label_data);    
    tradeoff_param=learner_param.tradeoff_param/e_num;

    sigma=learner_param.sigma;
    gamma=1/(2*sigma);


    svm_param=sprintf(' -c %f -t 2 -g %f -q', tradeoff_param, gamma);
    svm_model=libsvm_train(label_data, feat_data, svm_param);
    
    if svm_model.Label(1)<0
        svm_model.sv_coef=-svm_model.sv_coef;
        svm_model.rho=-svm_model.rho;
    end
        
    one_model_info=[];
    one_model_info.sv_indices=svm_model.sv_indices;
    one_model_info.sv_coef=svm_model.sv_coef;
    one_model_info.b=-svm_model.rho;
        
       
    learner_idx=bi_train_data.hash_learner_idx;
    hash_learners_model.model_infos{learner_idx}=one_model_info;
    hash_learner{1}=learner_idx;
    
    
    hash_learners_model.sv_sel(svm_model.sv_indices)=true;
          
    
    feat_code=do_predict_svm_rbf(bi_train_data.feat_data, svm_model);
    hlearner_bi_code=gen_bi_code_from_logical(feat_code);
       
    
    % debug:
%     hlearner_bi_code2=apply_svm_rbf(bi_train_data.feat_data, hash_learner, hash_learners_model);
%     assert(all(hlearner_bi_code==hlearner_bi_code2));

end



function feat_code=do_predict_svm_rbf(feat_data, svm_model)

    % assert(~issparse(feat_data));
    
    alpha=svm_model.sv_coef;

    if isempty(alpha)
        
        feat_code =false(size(feat_data,1),1);

    else


        gamma=svm_model.Parameters(4);
        
        svs=svm_model.SVs;	
        if ~issparse(feat_data)
        	svs=full(svs);	
        end

        b=-svm_model.rho;

        KTest = sqdist(feat_data',svs');
        KTest = exp(-KTest.*gamma);
        
        feat_code = KTest*alpha + b  > 0;

    end

end




function [hash_learners, hash_learners_model]=...
        do_post_process(hash_learners, hash_learners_model)
    
    
    sv_sel=hash_learners_model.sv_sel;
    hash_learners_model.support_vectors=hash_learners_model.support_vectors(sv_sel,:);
      
    sv_indices_map=zeros(length(sv_sel), 1);
    sv_indices_map(sv_sel)=1:size(hash_learners_model.support_vectors, 1);
    
    model_infos=hash_learners_model.model_infos;
    for b_idx=1:length(model_infos)
        one_model_info=model_infos{b_idx};
        one_model_info.sv_indices=sv_indices_map(one_model_info.sv_indices);
        model_infos{b_idx}=one_model_info;
    end
           
    hash_learners_model.model_infos=model_infos;
    
end



