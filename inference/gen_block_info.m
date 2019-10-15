% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com


function one_infer_info = gen_block_info(train_info, infer_info, g_idx)

% generate the unary and pairwise correspondence map according to the
% pairwise_relation_map in each block


trans_map=zeros(train_info.samples_num, 1, 'uint32');
r1_org_global = infer_info.pairwise_relation_map(:,1);
r2_org_global = infer_info.pairwise_relation_map(:,2);

shared_task_data=[];
shared_task_data.r1_org_global=r1_org_global;
shared_task_data.r2_org_global=r2_org_global;
shared_task_data.trans_map=trans_map;
shared_task_data.infer_block_info=infer_info.infer_block_info;
shared_task_data.e_num = train_info.samples_num;

    
one_infer_info = gen_one_group_info(g_idx, shared_task_data);
   

end



function one_infer_info = gen_one_group_info(g_idx, shared_task_data) 

      
     e_num = shared_task_data.e_num;
     r1_org_global=shared_task_data.r1_org_global;
     r2_org_global=shared_task_data.r2_org_global;
     trans_map=shared_task_data.trans_map;
     infer_block_info=shared_task_data.infer_block_info; 
     
     sel_e_idxes=infer_block_info{g_idx};
     sel_e_idxes=uint32(sel_e_idxes);
       
    
    r1_sel = ismember(r1_org_global, sel_e_idxes);
    r2_sel = ismember(r2_org_global, sel_e_idxes);
    
    r_sel = r1_sel | r2_sel;
    
    r1=r1_org_global(r_sel);  % get the corresponding sample idx 
    r2=r2_org_global(r_sel); 
    r1_sel=r1_sel(r_sel);    % get the corresponding sign(???r1_sel?????)
    r2_sel=r2_sel(r_sel);
    
    sel_r_idxes=uint32(find(r_sel)); % get the original pairs in the block
    
    % get the unique pairs
    
    original_matrix = cat(2, r1, r2);
    [unique_matrix, unique_rows, ~] = unique(original_matrix, 'rows');
    
    r1 = unique_matrix(:,1);
    r2 = unique_matrix(:,2);
    
    r1_sel = r1_sel(unique_rows);
    r2_sel = r2_sel(unique_rows);
    sel_r_idxes = sel_r_idxes(unique_rows);
    
    
    
    % delete the symmetric pairs in the relation map
    value_vector = ones(length(r1), 1);    
    block_pairwise_map = sparse(r1, r2, value_vector, e_num, e_num);
    total_idxes = sub2ind(size(block_pairwise_map), r1, r2);    
    
    block_pairwise_relation_map = block_pairwise_map + block_pairwise_map';
    upper_block_pairwise_relation_map = triu(block_pairwise_relation_map);
    [row_del, col_del] = find(upper_block_pairwise_relation_map == 2);
    del_idxes = sub2ind(size(upper_block_pairwise_relation_map), row_del, col_del);

    
    del_rows = find(ismember(total_idxes, del_idxes));
    sel_r_idxes(del_rows) = [];
    r1_sel(del_rows) = [];
    r2_sel(del_rows) = [];
    r1(del_rows) = [];
    r2(del_rows) = [];
             
    
 %   block_pairwise_map(del_idxes) = 0; 
  %  [r1, r2] = find(block_pairwise_map == 1);
    
             
        
    exchange_sel=~r1_sel;
    tmp_r1=r1(exchange_sel); 
    r1(exchange_sel)=r2(exchange_sel);
    r2(exchange_sel)=tmp_r1; 
    r2_sel(exchange_sel)=false;
    multual_sel=r2_sel; %multual_sel: +1:pairwise  0:unary
    
    trans_map(sel_e_idxes)=1:length(sel_e_idxes);
    
    
    r2_org=r2;
    r1=trans_map(r1);  %include the unary and pairwise parts, transfer the indexs of the training data to indexes of the block
    sel_r1=r1(multual_sel);
    sel_r2=r2(multual_sel);
    sel_r2=trans_map(sel_r2); %get the indexes the pairwise parts with "the indexes in block"
    
    sample_info=[];
    sample_info.multual_sel=multual_sel;
    
    non_sel1=~multual_sel;  %unary indexes
    sample_info.non_sel1=non_sel1;
    sample_info.non_sel2=[];
    sample_info.non_sel1_e_idxes=r1(non_sel1); %unary indexes of the block
    sample_info.non_sel1_e_idxes_other=r2_org(non_sel1); %indexes of the training data
    sample_info.non_sel2_e_idxes=[];
    sample_info.non_sel2_e_idxes_other=[];
    
    sample_info.sample_e_num=length(sel_e_idxes);
    sample_info.sel_r_idxes=sel_r_idxes; %the indexes of the pairs in the relation map that has the training sample in the block
        
    one_relation_map=cat(2, sel_r1, sel_r2); % get the pairwise indexes corresponding to the block
    assert(isa(one_relation_map, 'uint32'));
        
            
    one_infer_info=[];
    one_infer_info.sample_info=sample_info;
    one_infer_info.sel_e_idxes=sel_e_idxes;
    one_infer_info.e_num=length(sel_e_idxes);
    one_infer_info.relation_map=one_relation_map;
  

 end
