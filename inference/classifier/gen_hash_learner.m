

function [hash_learner hash_learners_model cache_info hlearner_bi_code]=gen_hash_learner(...
	hash_learner_param, bi_train_data, hash_learners_model, cache_info)


classifier_type=hash_learner_param.classifier_type;

gen_classifier_fn=[];

if strcmp(classifier_type, 'svm_linear')
	gen_classifier_fn=@gen_svm_linear;
end

if strcmp(classifier_type, 'svm_rbf')
	gen_classifier_fn=@gen_svm_rbf;
end

if strcmp(classifier_type, 'svm_rbf_budgeted')
	gen_classifier_fn=@gen_svm_rbf_budgeted;
end

if strcmp(classifier_type, 'svm_rbf_kernel_feat')
	gen_classifier_fn=@gen_svm_rbf_kernel_feat;
end


if strcmp(classifier_type, 'boost_tree')
	gen_classifier_fn=@gen_cdboost;
end


assert(~isempty(gen_classifier_fn));

[hash_learner hash_learners_model cache_info hlearner_bi_code]=gen_classifier_fn(...
    hash_learner_param, bi_train_data, hash_learners_model, cache_info);

assert(iscell(hash_learner));


end

