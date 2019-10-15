

% Code author: Guosheng Lin. Contact: guosheng.lin@gmail.com or guosheng.lin@adelaide.edu.au


function feat_data_code=tsh_encode(model, data_feat, bit_num)

fprintf('\n\n------------------------------fasthash_encode---------------------------\n\n');

if nargin<3
	bit_num=[];
end

if ~isempty(bit_num)
	model.hs=model.hs(1:bit_num, :);
end

feat_data_code=apply_hash_learner(data_feat, model.hs, model.hs_model);
feat_data_code=feat_data_code>0;

end




