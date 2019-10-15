

function bi_code=apply_svm_rbf_budgeted(feat_data, hash_learners, hash_learners_model)


parameter_string='';
label_data=ones(size(feat_data, 1), 1);


bit_num=size(hash_learners, 1);
bi_code=zeros(size(feat_data, 1), bit_num, 'int8');

for b_idx=1:bit_num
    
    model=hash_learners{b_idx}.model;
    [~, pred_labels] = budgetedsvm_predict(label_data, feat_data, model, parameter_string);

    feat_code=pred_labels>0;
    one_bi_code=gen_bi_code_from_logical(feat_code);

    bi_code(:, b_idx)=one_bi_code;
end



end
