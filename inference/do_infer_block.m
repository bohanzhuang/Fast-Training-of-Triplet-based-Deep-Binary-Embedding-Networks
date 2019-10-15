% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com


function infer_result = do_infer_block(train_info, infer_info)

infer_block_type=train_info.infer_block_type;
one_infer_fn=[];


if strcmp(infer_block_type, 'graphcut')
    one_infer_fn=@do_infer_graphcut; 
end

assert(~isempty(one_infer_fn));

infer_iter_num=train_info.infer_iter_num;
infer_iter_counter=0;

infer_bi_code=infer_info.init_bi_code; %initialize hash_codes all to 1

group_num=length(infer_info.infer_block_info);

%cost = []; %BQP cost
%triplet_loss = [];

% block descent
while true
    
    group_idxes=randperm(group_num); 
    
    for n=1:group_num
        
        g_idx=group_idxes(n);
        one_infer_info=gen_block_info(train_info, infer_info, g_idx);                      
        one_infer_info=update_infer_info_block(one_infer_info, infer_info, infer_bi_code, g_idx); 
        one_init_infer_result=[];
        one_init_infer_result.infer_bi_code=infer_bi_code(one_infer_info.sel_e_idxes);
        one_infer_result = one_infer_fn(one_infer_info, one_init_infer_result);
        one_bi_code=one_infer_result.infer_bi_code;
        infer_bi_code(one_infer_info.sel_e_idxes)=one_bi_code;
        
   %     one_cost = calculate_cost(infer_bi_code, infer_info);
   %      triplet_one_loss = calculate_triplet_cost(infer_bi_code, train_info);
   %      triplet_loss = [triplet_loss; triplet_one_loss];
   %     cost = [cost; one_cost];
        
        
    end
        infer_iter_counter=infer_iter_counter+1;
        
    if infer_iter_counter>=infer_iter_num
        break;
    end  
                
    
end

%save('cost', 'cost');
%save('triplet_loss', 'triplet_loss');

infer_result=gen_infer_result(['block_' infer_block_type], infer_bi_code);

end



function one_infer_info=update_infer_info_block(one_infer_info, infer_info, infer_bi_code, g_idx)
% get the unary and pairwise weights in the potential function

relation_weights=infer_info.pairwise_weights_vector;

[relation_weights, single_weights]=gen_relation_weight_block(...
        one_infer_info.sample_info, relation_weights, infer_bi_code, infer_info, g_idx);

one_infer_info.relation_weights=relation_weights(one_infer_info.sample_info.multual_sel);
one_infer_info.single_weights=single_weights;

end


function [relation_weights, single_weights]=gen_relation_weight_block(sample_info, relation_weights, init_bi_code, infer_info, g_idx)

sample_e_num=sample_info.sample_e_num;
relation_weights=relation_weights(sample_info.sel_r_idxes);

non_sel_bi_code=init_bi_code(sample_info.non_sel1_e_idxes_other);
    
non_sel_weights=relation_weights(sample_info.non_sel1).*double(non_sel_bi_code); 

non_sel_weights = full(non_sel_weights);
        
single_weights = idxsum(non_sel_weights, sample_info.non_sel1_e_idxes, sample_e_num);


end

%function single_weights_2 = get_unary_weights(infer_info, g_idx)

%group_idxes = infer_info.infer_block_info{g_idx};

%single_weights_2 = infer_info.unary_weights(group_idxes);

%end



function infer_result=do_infer_graphcut(infer_info, init_infer_result)


e_num=infer_info.e_num;
relation_map=infer_info.relation_map;
relation_weights=infer_info.relation_weights;

if max(relation_weights)>eps
    % dbstack;
    % keyboard;

    fprintf('\n WARNING, submodularity is not satisfied...\n');
    relation_weights=min(relation_weights, 0);
end

if ~isa(relation_map,'double')
    relation_map=double(relation_map);
end

weight_mat_block = sparse(relation_map(:,1), relation_map(:,2), -relation_weights,e_num,e_num);

weight_mat_block=weight_mat_block+weight_mat_block'; %make it symmetric

single_weights=infer_info.single_weights';

assert(~isempty(single_weights));

unary = cat(1, zeros(1, e_num), single_weights); %notice here!!!

label_pairwise_cost=ones(2,2);
label_pairwise_cost(1,1)=0;
label_pairwise_cost(2,2)=0;

init_label=zeros(1, e_num);
init_label(init_infer_result.infer_bi_code>0)=1;

if nnz(weight_mat_block)>0  % the graph-cuts infer function
    labels = GCMex(init_label, single(unary), weight_mat_block, single(label_pairwise_cost));
else
    [~, labels]=min(unary, [], 1);
    labels=labels-1;
end


infer_bi_code=ones(length(labels),1);
infer_bi_code(labels<1)=-1;
infer_bi_code=infer_bi_code(1:infer_info.e_num);

infer_result=[];
infer_result.infer_bi_code=infer_bi_code;


end



function sum_v=idxsum(values, idxes, value_num)

sum_v=accumarray(idxes,values);

if length(sum_v)<value_num
    sum_v(value_num)=0;
end

end




function infer_result=gen_infer_result(infer_name, infer_bi_code)


infer_result=[];
infer_result.infer_name=infer_name;
infer_result.infer_bi_code=int8(infer_bi_code);


end

