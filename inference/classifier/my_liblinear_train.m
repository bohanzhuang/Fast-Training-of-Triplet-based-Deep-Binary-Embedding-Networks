
function svm_model = my_liblinear_train(label_data, feat_data, params);

% TODO: 
% use weighted linear SVM:
%     data_weight=bi_train_data.data_weight;
%     if isempty(data_weight)
%         svm_model = liblinear_train(label_data, feat_data, params);
%     else
%         svm_model = liblinear_train_weight(data_weight, label_data, feat_data, params);
%     end


	assert(~issparse(label_data));
    assert(isa(label_data, 'double'));

    if issparse(feat_data)
    	
    	% for sparse data:
    	assert(isa(feat_data, 'double'));
    	svm_model = liblinear_train(label_data, feat_data, params);

    else

    	% for dense data:
    	assert(isa(feat_data, 'single'));
    	svm_model = liblinear_dense_float_train(label_data, feat_data, params);

    end



end