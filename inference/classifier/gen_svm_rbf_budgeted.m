

function [hash_learner hash_learners_model cache_info hlearner_bi_code]=gen_svm_rbf_budgeted(...
    learner_param, bi_train_data, hash_learners_model, cache_info)

    if ~isfield(cache_info, 'is_init') || ~cache_info.is_init
                
        hash_learners_model.apply_hash_learner_fn=@apply_svm_rbf_budgeted;
        cache_info.is_init=true;
        
    end
    

%     
% Parameter string is of the same format for both versions, specified as follows:
% 
% 	'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'
% 
% 	The following options are available (default values in parentheses):
% 	A - algorithm, which large-scale SVM to use (2):
% 	    0 - Pegasos
% 	    1 - AMM batch
% 	    2 - AMM online
% 	    3 - LLSVM
% 	    4 - BSGD
% 	d - dimensionality of the data (required when inputs are .txt files, in that case MUST be set by a user)
% 	e - number of training epochs in AMM and BSGD (5)
% 	s - number of subepochs in AMM batch (1)
% 	b - bias term in AMM, if 0 no bias added (1.0)
% 	k - pruning frequency, after how many observed examples is pruning done in AMM (10000)
% 	C - pruning aggresiveness, sets the pruning threshold in AMM, OR
% 			linear-SVM regularization paramater C in LLSVM (10.0)
% 	l - limit on the number of weights per class in AMM (20)
% 	L - learning parameter lambda in AMM and BSGD (0.00010)
% 	G - kernel width exp(-0.5 * gamma * ||x_i - x_j||^2) in BSGD and LLSVM (1 / DIMENSIONALITY)
% 	B - total SV set budget in BSGD, OR number of landmard points in LLSVM (100)
% 	M - budget maintenance strategy in BSGD (0 - removal; 1 - merging), OR
% 			landmark sampling strategy in LLSVM (0 - random; 1 - k-means; 2 - k-medoids) (1)
% 	
% 	z - training and test file are loaded in chunks so that the algorithm can 
% 			handle budget files on weaker computers; z specifies number of examples loaded in
% 			a single chunk of data, ONLY when inputs are .txt files (50000)
% 	w - model weights are split in chunks, so that the algorithm can handle
% 		highly dimensional data on weaker computers; w specifies number of dimensions stored
% 		in one chunk, ONLY when inputs are .txt files (1000)
% 	S - if set to 1 data is assumed sparse, if 0 data assumed non-sparse, used to
% 			speed up kernel computations (default is 1 when percentage of non-zero
% 		    features is less than 5%, and 0 when percentage is larger than 5%)
% 	v - verbose output: 1 to show the algorithm steps (epoch ended, training started, ...), 0 for quiet mode (0)	
% 	--------------------------------------------


    sigma=learner_param.sigma;
    gamma=1/sigma;
    
    epoch_num=learner_param.epoch_num;
    budget=learner_param.budget;
    
    
    label_data=bi_train_data.label_data;
    feat_data=bi_train_data.feat_data;
    
    e_num=length(label_data);    
    tradeoff_param=learner_param.tradeoff_param/e_num;
    lambda=0.5/tradeoff_param;
    
    

    % assert(~issparse(feat_data));
    assert(isa(feat_data, 'double'));
    
    assert(~issparse(label_data));
    assert(isa(label_data, 'double'));
    
       
    alg_param=4;
    data_sparse=issparse(feat_data);
    
    parameter_string=sprintf('-b 1 -A %d -e %d -L %f -G %f -B %d -S %d',...
        alg_param, epoch_num, lambda, gamma, budget, data_sparse);
    
    model = budgetedsvm_train(label_data, feat_data, parameter_string);
        
           
    hash_learner{1}.model=model;
    
    hlearner_bi_code=apply_svm_rbf_budgeted(feat_data, hash_learner, hash_learners_model);

end

