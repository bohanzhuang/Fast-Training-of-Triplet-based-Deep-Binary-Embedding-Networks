function infer_block_info = gen_infer_block(pairwise_coefficients_matrix, train_info)

fprintf('\n------------- gen_infer_block... \n')

relevant_groups=[];
if isempty(relevant_groups)
     relevant_groups=gen_relevant_groups(pairwise_coefficients_matrix, train_info);
end
    
   assert(~isempty(relevant_groups));   
    
 fprintf('\n------------- gen_infer_block finished \n')


infer_block_info = relevant_groups;
end


function relevant_groups = gen_relevant_groups(pairwise_coefficients_matrix, train_info)



disp_step=0.05;
disp_thresh=disp_step;
e_num= train_info.samples_num;
relevant_groups=cell(0,1);
can_sel=true(e_num,1);

finish_rate = 0;

while finish_rate<1
   
    group_e_idxes=gen_one_group(pairwise_coefficients_matrix, can_sel);
    can_sel(group_e_idxes)=false; %备选集
    relevant_groups=cat(1, relevant_groups, {group_e_idxes});
    finish_rate=(e_num-nnz(can_sel))/e_num;
     
    if finish_rate>=disp_thresh
        fprintf(' %.2f ', finish_rate);
        disp_thresh=disp_thresh+disp_step;
    end
      
    
    
end
  fprintf(' <--done!\n');
 

end


function group_e_idxes = gen_one_group(pairwise_coefficients_matrix, can_sel)

group_e_sel=false(length(can_sel), 1);
can_e_idxes=find(can_sel);

root_e_idx=can_e_idxes(randsample(length(can_e_idxes), 1));

group_e_sel(root_e_idx)=true;
can_sel(root_e_idx)=false;

rel_e_idxes = get_relevant_idxes(pairwise_coefficients_matrix, root_e_idx);
can_sel(rel_e_idxes)=false;

can_e_idxes=find(can_sel);
can_e_idxes=can_e_idxes(randperm(length(can_e_idxes)));

can_e_idxes=cat(1, rel_e_idxes', can_e_idxes);

for g_idx=1:length(can_e_idxes)
    other_e_idx=can_e_idxes(g_idx);
    is_valid=check_consistent(pairwise_coefficients_matrix, group_e_sel, other_e_idx);
    
     if is_valid
        group_e_sel(other_e_idx)=true;
     end    
    
end

group_e_idxes=find(group_e_sel);


end

function rel_e_idxes = get_relevant_idxes(pairwise_coefficients_matrix, root_e_idx)

pairwise_coefficients_matrix(root_e_idx, root_e_idx) = 1;
rel_e_idxes = find(pairwise_coefficients_matrix(root_e_idx,:)<0);


end

function is_valid = check_consistent(pairwise_coefficients_matrix, group_e_sel, other_e_idx)

idxes = find(group_e_sel);
other_e_idxes_expand = repmat(other_e_idx, [length(idxes), 1]);

idx_list = linspace(1, length(idxes), length(idxes))';

check_coefficients_list = pairwise_coefficients_matrix(sub2ind(size(pairwise_coefficients_matrix), idxes, other_e_idxes_expand(idx_list)));

is_valid=isempty(find(check_coefficients_list>0));

end



