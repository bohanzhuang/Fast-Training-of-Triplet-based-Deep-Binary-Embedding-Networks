

function bi_code=apply_cdboost(feat_data, hash_learners, hash_learners_model)
    
   
   	test_data=[];
   	test_data.feat_data=feat_data;

   	e_num=size(feat_data, 1);
   	bit_num=length(hash_learners);
   	feat_code=false(e_num, bit_num);
   	for h_idx=1:bit_num
   		model=hash_learners{h_idx};
   		predict_result=cdboost_predict(model, test_data, []);
   		one_feat_code=predict_result.predict_labels>0;
   		feat_code(:, h_idx)=one_feat_code;
   	end

    
    bi_code=gen_bi_code_from_logical(feat_code);
      
end
