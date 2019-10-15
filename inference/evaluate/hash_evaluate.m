


function [predict_result]=hash_evaluate(eva_param, code_data_info)

disp('hash_evaluate...');


dist_info=gen_dist_info(code_data_info.tst_data_code, code_data_info.db_data_code, eva_param);

db_label_info=code_data_info.db_label_info;
test_label_info=code_data_info.test_label_info;

cache_info=[];
cache_info=calc_agree_mat(cache_info, dist_info, db_label_info, test_label_info);

result_ir=[]; 
result_ir=do_eva_pk(eva_param, result_ir, cache_info);

predict_result=result_ir;

end




function [cache_info]=calc_agree_mat(cache_info, dist_info, db_label_info, test_label_info)
	
    dist_sort_idx_mat=dist_info.dist_sort_idx_mat;
    
    label_type=db_label_info.label_type;
    
    agree_mat=[];      

            
    if strcmp(label_type, 'multiclass')
                    
        Ytrain=db_label_info.label_data;
        Ytest=test_label_info.label_data;

        assert(length(unique(Ytrain))<2^16);
        Ytrain=uint16(Ytrain);
        Ytest=uint16(Ytest);
                                                
        Labels  = Ytrain(dist_sort_idx_mat);
        if size(dist_sort_idx_mat,1)==1
            Labels=Labels';
        end
        agree_mat   = bsxfun(@eq, Ytest, Labels); 
        
        cache_info.test_data_labels=Ytest;
        cache_info.train_data_labels=Ytrain;
    end
    
    if strcmp(label_type, 'multilabel')
        
        train_multilabel =db_label_info.label_data;
        test_multilabel =test_label_info.label_data;
        assert(length(unique(train_multilabel))<2^16);
        train_multilabel=uint16(train_multilabel);
        test_multilabel=uint16(test_multilabel);
        
        agree_mat = zeros(size(test_multilabel, 1), size(dist_sort_idx_mat, 2));
        
        for i = 1:size(test_multilabel, 1)
           
            sort_idx = dist_sort_idx_mat(i,:)';
            label_map = train_multilabel(sort_idx,:);
            test_map = test_multilabel(i,:);
            
            sub_agree_mat = sum(bsxfun(@times, test_map, label_map), 2);
            sub_agree_mat(find(sub_agree_mat>0))  = 1;
            agree_mat(i,:) = sub_agree_mat';
            
        end
            
        
    end
    
    
    
    %NOTES: user can add their code here for evaluation based on their application
%     if strcmp(label_type, 'custom')
%         
%     end
          
    assert(~isempty(agree_mat));
    cache_info.agree_mat=agree_mat;
    
end







function result_ir=do_eva_pk(eva_param, result_ir, cache_info)


agree_mat=cache_info.agree_mat;
train_num = size(agree_mat, 2);
test_num  = size(agree_mat, 1);
test_k=eva_param.eva_top_knn_pk;
test_k = min(test_k, train_num);

if strcmp(eva_param.test_method, 'precision')

eva_name=['pk' num2str(test_k)];

result_ir.(eva_name)   = mean( mean( agree_mat(:, 1:test_k), 2 ) );
else
    
agree_mat_map=agree_mat(:, 1:test_k);
tmp_cumsum=cumsum(agree_mat_map, 2);   
tmp_mAP = bsxfun(@ldivide, (1:test_k), tmp_cumsum);
tmp_sum=sum(agree_mat_map, 2);
tmp_sum=max(tmp_sum,1);    
    
map = mean(sum(tmp_mAP .* agree_mat_map, 2)./ tmp_sum);
eva_name = ['map' num2str(test_k)];
result_ir.(eva_name) = map;
    
    
end



end



   




function dist_info=gen_dist_info(code_1, code_2, eva_param)


if ~isfield(eva_param, 'use_weight_hamming')
    eva_param.use_weight_hamming=false;
end

if eva_param.use_weight_hamming
    calc_hamming_dist_fn=@calc_hamming_dist_weight;
else
    calc_hamming_dist_fn=@calc_hamming_dist;
end


trn_num=size(code_2,1);
test_num=size(code_1,1);

max_knn=1e4;
max_knn=min(max_knn, trn_num);

large_data_thresh=5e3;
eva_capacity=ceil(large_data_thresh^2);

dist_mat=[];

if trn_num*test_num>eva_capacity

    assert(trn_num<2^31);
    max_bit_num=size(code_1,2)*8;
    assert(max_bit_num<2^15);

    dist_sort_idx_mat=zeros(test_num, max_knn, 'uint32');
    sort_dist_mat=zeros(test_num, max_knn, 'uint16');

    trn_step_size=min(trn_num, eva_capacity);
    tst_step_size=min(test_num,ceil(eva_capacity/trn_step_size));


    tst_e_counter=0;


    while tst_e_counter<test_num
        
        tst_start_idx=tst_e_counter+1;
        tst_end_idx=tst_e_counter+tst_step_size;
        tst_end_idx=min(test_num, tst_end_idx);
        tst_sel_idxes=tst_start_idx:tst_end_idx;


        one_test_num=length(tst_sel_idxes);
        sel_test_one_dist=zeros(one_test_num, trn_num, 'uint16');


        step_size=trn_step_size;
        e_counter=0;

        sel_code_1_tmp=code_1(tst_sel_idxes, :);

        while e_counter<trn_num
            start_idx=e_counter+1;
            end_idx=e_counter+step_size;
            end_idx=min(trn_num, end_idx);
            sel_idxes=start_idx:end_idx;

            sel_code_2_tmp=code_2(sel_idxes, :);

            if size(sel_code_1_tmp,1)<size(sel_code_2_tmp,1)
                one_one_dist  = calc_hamming_dist_fn(sel_code_1_tmp, sel_code_2_tmp, eva_param);
            else
                one_one_dist  = calc_hamming_dist_fn(sel_code_2_tmp, sel_code_1_tmp, eva_param);
                one_one_dist=one_one_dist';
            end

            sel_test_one_dist(:, sel_idxes)=uint16(one_one_dist);
            e_counter=end_idx;

        end


        tst_e_counter=tst_end_idx;
        [one_sort_dist_mat,one_I]   = sort(sel_test_one_dist, 2);

        one_I=one_I(:, 1:max_knn);
        one_sort_dist_mat=one_sort_dist_mat(:, 1:max_knn);

        dist_sort_idx_mat(tst_sel_idxes,:)=uint32(one_I);
        sort_dist_mat(tst_sel_idxes,:)=uint16(one_sort_dist_mat);

    end

    

else

    dist_mat  = calc_hamming_dist_fn(code_1, code_2, eva_param);
    dist_mat = uint16(dist_mat);
    [sort_dist_mat,dist_sort_idx_mat]   = sort(dist_mat, 2);
    
    if size(dist_sort_idx_mat, 2)>max_knn
        dist_sort_idx_mat=dist_sort_idx_mat(:, 1:max_knn);
        sort_dist_mat=sort_dist_mat(:, 1:max_knn);
    end

end
    

    
dist_info.dist_mat=dist_mat;
dist_info.dist_sort_idx_mat=dist_sort_idx_mat;
dist_info.sort_dist_mat=sort_dist_mat;


end






function dist=calc_hamming_dist(code_1, code_2, eva_param)


assert(islogical(code_1)); 
assert(islogical(code_2));

e_num1=size(code_1,1);
e_num2=size(code_2,1);


    assert(size(code_1,2)<2^15);
    dist=zeros(e_num1, e_num2, 'uint16');
    
    for e_ind=1:e_num1
        one_pair_feat=bsxfun(@xor, code_1(e_ind,:), code_2);
        assert(islogical(one_pair_feat));
        one_dist=sum(one_pair_feat, 2);
        dist(e_ind,:)=uint16(one_dist);
    end

end


function dist=calc_hamming_dist_weight(code_1, code_2, eva_param)


assert(islogical(code_1)); 
assert(islogical(code_2));

e_num1=size(code_1,1);
e_num2=size(code_2,1);


w=eva_param.hamming_weight;

    
dist=zeros(e_num1, e_num2, 'single');

for e_ind=1:e_num1
    one_pair_feat=bsxfun(@xor, code_1(e_ind,:), code_2);
    if length(w)>size(one_pair_feat,2)
        w=w(1:size(one_pair_feat,2));
    end
    one_dist=one_pair_feat*w;
    dist(e_ind,:)=single(one_dist);
end
    

end






