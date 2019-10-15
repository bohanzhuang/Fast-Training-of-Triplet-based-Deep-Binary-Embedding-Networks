
function predict_result=cdboost_predict(model, test_data, predict_config)

w=model.w;
tree_models=model.wl_model.tree_models;
iter_num=length(w);

feat_data=test_data.feat_data;
e_num=size(feat_data, 1);


sel_feat_idxes=model.sel_feat_idxes;
if ~isempty(sel_feat_idxes)
    feat_data=feat_data(:, sel_feat_idxes);
end


predict_scores=zeros(e_num, 1);
for b_idx=1:iter_num
    
    one_tree_model=tree_models{b_idx};
    hfeat=one_apply_wl(feat_data, one_tree_model);
    one_scores=hfeat.*w(b_idx);
    predict_scores=predict_scores+one_scores;
end

predict_labels=nonzerosign(predict_scores);

predict_result=[];
predict_result.predict_labels=predict_labels;
predict_result.predict_scores=predict_scores;


end


function hfeat=one_apply_wl(feat_data, one_model)

sel_feat_idxes=one_model.sel_feat_idxes;
if ~isempty(sel_feat_idxes)
    feat_data=feat_data(:, sel_feat_idxes);
end
hfeat = binaryTreeApply( feat_data, one_model);
hfeat= nonzerosign(hfeat);


end
