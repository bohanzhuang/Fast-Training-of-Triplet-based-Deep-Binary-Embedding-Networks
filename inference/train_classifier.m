function  work_info_step2=train_classifier(train_info, train_data, work_info_step2)


update_bit=work_info_step2.update_bit;
hash_learner_cache_info=work_info_step2.hash_learner_cache_info;

bi_train_data=[];
bi_train_data.feat_data=train_data.feat_data;
bi_train_data.label_data=double(work_info_step2.update_bi_code);
bi_train_data.hash_learner_idx=update_bit;
bi_train_data.data_weight = work_info_step2.data_weights;


hash_learners_model=work_info_step2.hash_learners_model;

[hash_learner_info, hash_learners_model, hash_learner_cache_info, hlearner_bi_code]=...
    train_hash_learner(train_info, bi_train_data, hash_learners_model, hash_learner_cache_info);


work_info_step2.hash_learners_model=hash_learners_model;
work_info_step2.hash_learner_infos{update_bit}=hash_learner_info;
work_info_step2.hash_learner_cache_info=hash_learner_cache_info;
work_info_step2.update_bi_code_step2=hlearner_bi_code;



end


function [hash_learner_info, hash_learners_model, cache_info, hlearner_bi_code]=train_hash_learner(...
    train_info, bi_train_data, hash_learners_model, cache_info)

feat_data=bi_train_data.feat_data;
e_num=size(feat_data, 1);

hash_learner_param=train_info.hash_learner_param;

[one_hash_learner, hash_learners_model, cache_info, hlearner_bi_code]=gen_hash_learner(...
    hash_learner_param, bi_train_data, hash_learners_model, cache_info);

assert(length(hlearner_bi_code)==e_num);

hash_learner_info=[];
hash_learner_info.hash_learner=one_hash_learner;

pos_sel=bi_train_data.label_data>0;
pos_num=nnz(pos_sel);

hash_learner_info.e_num=e_num;
hash_learner_info.pos_num=pos_num;
hash_learner_info.neg_num=hash_learner_info.e_num-hash_learner_info.pos_num;
hash_learner_info.use_data_weight=~isempty(bi_train_data.data_weight);

%return accuracy and weighted accuracy
hash_learner_info.acc = calc_accuracy(hlearner_bi_code, bi_train_data.label_data);


end


function acc = calc_accuracy(predict_labels, gt_labels)

    correct_sel=gt_labels == predict_labels;
    acc=nnz(correct_sel)./length(correct_sel);


end