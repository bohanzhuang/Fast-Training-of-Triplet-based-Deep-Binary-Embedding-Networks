

% author: Guosheng Lin
% contact: guosheng.lin@gmail.com



function train_result=cdboost_train(train_info, train_data)



e_num=size(train_data.feat_data,1);
feat_num=size(train_data.feat_data,2);



t0=tic;


max_wl_num=200;
if isfield(train_info, 'max_wl_num')
    max_wl_num=train_info.max_wl_num;
end

max_iteration_num=max_wl_num;

train_info.one_tree_mode=true;
if max_iteration_num>1
    train_info.one_tree_mode=false;
end

%------------------------------------------------------

% sampling setting

if ~isfield(train_info, 'do_mass_selection')
    train_info.do_mass_selection=true;
    if train_info.one_tree_mode
        train_info.do_mass_selection=false;
    end
end

if train_info.do_mass_selection
    if ~isfield(train_info, 'mass_weight_thresh')
        train_info.mass_weight_thresh=0.99;
    end
else
    train_info.mass_weight_thresh=1;
end


if ~isfield(train_info, 'tree_node_feat_num')
    
    train_info.tree_node_feat_num=200;
%     train_info.tree_node_feat_num=inf;
    if train_info.one_tree_mode
        train_info.tree_node_feat_num=inf;
    end

end



if ~isfield(train_info, 'tree_node_e_num')
    train_info.tree_node_e_num=inf;
end



if ~isfield(train_info, 'tree_depth')
    train_info.tree_depth=4;
end

if ~isfield(train_info, 'thread_num')
    train_info.thread_num=8;
end









%------------------------------------------------------



% sovler config

if ~isfield(train_info,'use_stagewise')
    train_info.use_stagewise=true;
end


if ~isfield(train_info, 'tradeoff_nv')
    train_info.tradeoff_nv=1e-8;
end



%------------------------------------------------------









label_data=train_data.label_data;
feat_data=train_data.feat_data;
data_weight=train_data.data_weight;


if isempty(data_weight)
    data_weight=ones(size(label_data));
end


try
assert(size(label_data, 1)==size(feat_data, 1));
assert(size(label_data, 1)==size(data_weight, 1));
catch err_info
    disp(err_info);
    dbstack;
    keyboard;
end



work_info=[];
work_info.pair_label_losses=data_weight;
work_info.pair_feat=zeros(size(feat_data, 1), max_wl_num, 'int8');
work_info.wl_model_infos=cell(max_wl_num, 1);
work_info.solver_wl_valid_sel=false(max_wl_num, 1);


wl_model=[];
wl_model.tree_models=cell(max_wl_num, 1);

solver_result=[];
solver_t=0;
wl_t=0;


wl_data_weight=data_weight;



feat_data_new=[];
feat_data_new.pos_sel=label_data>0;
feat_data_new.neg_sel=~feat_data_new.pos_sel;
feat_data_new.tree_data.X1=feat_data(feat_data_new.pos_sel,:);
feat_data_new.tree_data.X0=feat_data(feat_data_new.neg_sel,:);
feat_data_new.feat_num=size(feat_data,2);
feat_data_new.e_num=size(feat_data,1);

feat_data=feat_data_new;


for iter_idx=1:max_iteration_num
            
    
        
    wl_idx=iter_idx;
    work_info.wl_model_infos{wl_idx}=gen_wl_model_info(wl_idx);            
    
    
    t1=tic;
    [wl_model work_info]=one_gen_wl(train_info, wl_model, feat_data, label_data, wl_data_weight, work_info, wl_idx);
    wl_t=wl_t+toc(t1);
    
        
    work_info.solver_wl_valid_sel(wl_idx)=true;
    work_info.solver_feat_change_idxes=wl_idx;
    work_info.solver_update_wl_idxes=wl_idx;
            
    t2=tic;
        
    if train_info.one_tree_mode
        solver_result.method='one_tree';
        solver_result.w=1;
    else
        [solver_result work_info]=cdboost_solver(train_info, work_info);
        wl_data_weight=solver_result.wlearner_pair_weight;
        
        wl_data_weight=(length(wl_data_weight)/sum(wl_data_weight)).*wl_data_weight;
    end
    
    solver_t=solver_t+toc(t2);
    
    wl_num=length(solver_result.w);
    
    assert(wl_num==length(wl_model.tree_models));
    assert(wl_num==size(work_info.pair_feat, 2));
       

end


model_w=solver_result.w;
wl_sel=model_w>0;

if nnz(wl_sel)==0
    wl_sel(1)=true;
end

failed_wl_num=length(wl_sel)-nnz(wl_sel);

if failed_wl_num>0
    wl_model.tree_models=wl_model.tree_models(wl_sel);
    model_w=model_w(wl_sel);
end


model=[];
model.w=model_w;
model.wl_model=wl_model;
model.sel_feat_idxes=[];


total_t=toc(t0);

train_result=[];
train_result.model=model;
train_result.train_tim=total_t;

wl_num=length(model.w);
sel_feat_num=0;
sel_e_num=0;
sel_e_t=0;
sel_feat_t=0;
gen_wl_t=0;
apply_wl_t=0;
feat_mining_t=0;
tree_e_sel_time=0;
tree_train_time=0;
tree_other_time=0;
tree_total_time=0;
tree_max_feat_num=0;
tree_max_e_num=0;
mean_confidence=0;
for wl_idx=1:wl_num
    one_model_info=work_info.wl_model_infos{wl_idx};
    sel_feat_num=sel_feat_num+one_model_info.sel_feat_num;
    sel_e_num=sel_e_num+one_model_info.sel_e_num;
    sel_e_t=sel_e_t+one_model_info.sel_e_t;
    sel_feat_t=sel_feat_t+one_model_info.sel_feat_t;
    gen_wl_t=gen_wl_t+one_model_info.gen_wl_t;
    apply_wl_t=apply_wl_t+one_model_info.apply_wl_t;
    feat_mining_t=feat_mining_t+one_model_info.feat_mining_t;
    
    one_model=wl_model.tree_models{wl_idx};
    tree_e_sel_time=tree_e_sel_time+one_model.e_sel_time;
    tree_train_time=tree_train_time+one_model.train_time;
    tree_other_time=tree_other_time+one_model.other_time;
    tree_total_time=tree_total_time+one_model.total_time;
    tree_max_e_num=tree_max_e_num+one_model.max_e_num;
    tree_max_feat_num=tree_max_feat_num+one_model.max_feat_num;
    mean_confidence=mean_confidence+one_model.mean_confidence;
end
sel_e_num=ceil(sel_e_num/wl_num);
sel_feat_num=ceil(sel_feat_num/wl_num);
tree_max_feat_num=ceil(tree_max_feat_num/wl_num);
tree_max_e_num=ceil(tree_max_e_num/wl_num);
mean_confidence=mean_confidence/wl_num;

try
tree_depth=wl_model.tree_models{1}.depth;
catch err_info
    disp(err_info);
    dbstack;
    keyboard;
end


method_name=solver_result.method;

fprintf('\n----tree_learn, sel_feat:%d(%d), sel_e:%d, time: e_sel:%.1f, train:%.1f, other:%.1f, toatl:%.1f, depth:%d\n', ...
    tree_max_feat_num, feat_num, tree_max_e_num, tree_e_sel_time, tree_train_time, tree_other_time, tree_total_time, tree_depth);

fprintf('----cdboost, wl_num:%d(%d, failed:%d), w_trim:%.2f, sel_e:%d(%d), stage:%d, sol_name:%s\n', ...
    wl_num, max_iteration_num, failed_wl_num, train_info.mass_weight_thresh, sel_e_num, e_num,...
    train_info.use_stagewise, method_name);

fprintf('----cdboost, conf:%.4f, time: sel_e:%.1f, sel_f:%.1f, wl_gen:%.1f, wl_app:%.1f, solver:%.1f, wl:%.1f, total:%.1f\n', ...
    mean_confidence, sel_e_t, sel_feat_t, gen_wl_t, apply_wl_t, solver_t, wl_t, total_t);

fprintf('\n');

end




function [wl_model work_info]=one_gen_wl(wl_param, wl_model, feat_data_input,...
    label_data_input, data_weight_input, work_info, wl_idx)

one_model_info=work_info.wl_model_infos{wl_idx};


one_model_info=wl_sel_e(wl_param, one_model_info, label_data_input, data_weight_input);

wl_param.wl_idx=wl_idx;
[one_model one_model_info]=do_one_gen_wl(wl_param, feat_data_input,...
    label_data_input, data_weight_input, one_model_info);

tmp_t=tic;



[hfeat mean_confidence]=one_apply_wl(feat_data_input, one_model);


one_model.mean_confidence=mean_confidence;

one_model_info.apply_wl_t=one_model_info.apply_wl_t+toc(tmp_t);
one_pair_feat=label_data_input.*hfeat;
work_info.pair_feat(:, wl_idx)=one_pair_feat;
wl_model.tree_models{wl_idx}=one_model;

work_info.wl_model_infos{wl_idx}=one_model_info;



end





function [one_model one_model_info]=do_one_gen_wl(wl_param, feat_data_input,...
    label_data_input, data_weight_input, one_model_info)
        
        
    [feat_data, label_data, data_weight, one_model_info]=gen_wl_trn_data(...
        feat_data_input, label_data_input, data_weight_input, one_model_info);
    
   
    one_model_info.sel_e_num=feat_data.sel_e_num;
    one_model_info.sel_feat_num=feat_data.feat_num;
    

    tmp_t=tic;
    [one_model]=wl_train(wl_param, feat_data, label_data, data_weight);
    one_model_info.gen_wl_t=one_model_info.gen_wl_t+toc(tmp_t);
    one_model.sel_feat_idxes=one_model_info.sel_feat_idxes;
       
    
    one_tree_sel_feat_idxes_sub=one_model.fids+1;
    one_tree_sel_feat_idxes_sub=unique(one_tree_sel_feat_idxes_sub);
    if isempty(one_model_info.sel_feat_idxes)
        one_tree_sel_feat_idxes=one_tree_sel_feat_idxes_sub;
    else
        one_tree_sel_feat_idxes=one_model_info.sel_feat_idxes(one_tree_sel_feat_idxes_sub);
    end
    one_model_info.one_tree_sel_feat_idxes=one_tree_sel_feat_idxes;
    
  
    
end









function [one_model]=wl_train(wl_param, feat_data, label_data, data_weight)




tree_depth=wl_param.tree_depth;
nThreads=wl_param.thread_num;
    
 
    
pTree=[];
pTree.maxDepth=tree_depth;
pTree.nThreads=nThreads;


max_feat_num=wl_param.tree_node_feat_num;
feat_num=feat_data.feat_num;
if max_feat_num<feat_num
    fracFtrs=max_feat_num./feat_num;
    fracFtrs=min(fracFtrs, 1);
    pTree.fracFtrs=fracFtrs;
end



tree_data=feat_data.tree_data;

assert(isa(tree_data.X0, 'uint8'));
assert(isa(tree_data.X1, 'uint8'));

pos_sel=feat_data.pos_sel;
neg_sel=feat_data.neg_sel;

assert( ~isempty(data_weight));
tree_data.wts1=data_weight(pos_sel);
tree_data.wts0=data_weight(neg_sel);


if ~isempty(feat_data.e_sel)
  pTree.e_sel_X0=feat_data.e_sel(feat_data.neg_sel);
  pTree.e_sel_X1=feat_data.e_sel(feat_data.pos_sel);
else

  pTree.e_sel_X0=tree_data.wts0>0;
  pTree.e_sel_X1=tree_data.wts1>0;
end


try
assert( ~isempty(pTree.e_sel_X0));
assert( ~isempty(pTree.e_sel_X1));
catch err_info
    dbstack;
    keyboard;
end




max_e_num=wl_param.tree_node_e_num;
pTree.max_e_num=max_e_num;


one_model = binaryTreeTrain_sample( tree_data, pTree );


one_model.max_feat_num=max_feat_num;
one_model.max_e_num=max_e_num;
one_model.depth=tree_depth;




end







function [hfeat mean_confidence]=one_apply_wl(feat_data, one_model)



sel_feat_idxes=one_model.sel_feat_idxes;


assert(isempty(sel_feat_idxes));

pos_hfeat = binaryTreeApply( feat_data.tree_data.X1, one_model);
neg_hfeat = binaryTreeApply( feat_data.tree_data.X0, one_model);

hfeat=zeros(feat_data.e_num,1);
hfeat(feat_data.pos_sel)=pos_hfeat;
hfeat(feat_data.neg_sel)=neg_hfeat;

mean_confidence=mean(abs(hfeat));

hfeat= nonzerosign(hfeat);





end









function mass_e_sel=do_sel_e_mass(data_weight, weight_mass_thresh)


% this function only do mass selection

assert( weight_mass_thresh<1);

[tmp_weights, tmp_sel_e_idxes]=sort(data_weight, 'descend');
tmp_cumsum=cumsum(tmp_weights);
tmp_cumsum=tmp_cumsum./(tmp_cumsum(end)+eps);
tmp_pick_idx=find(tmp_cumsum>weight_mass_thresh, 1);
mass_sel_e_idexes=tmp_sel_e_idxes(1:tmp_pick_idx);

            
e_num=length(data_weight);
mass_e_sel=false(e_num,1);
mass_e_sel(mass_sel_e_idexes)=true;


end




function one_model_info=wl_sel_e(wl_param, one_model_info, label_data, data_weight)


%---------------------------- do select examples-----------------
    
    tmp_t=tic;
        
    e_sel=[];
    if wl_param.do_mass_selection
        
        mass_weight_thresh=wl_param.mass_weight_thresh;
        [e_sel]=do_sel_e_mass(data_weight, mass_weight_thresh);
    end

    one_model_info.e_sel=e_sel;
    one_model_info.sel_e_t=one_model_info.sel_e_t+toc(tmp_t);

%---------------------------- do select examples end-----------------


end






function [feat_data, label_data, data_weight, one_model_info]=gen_wl_trn_data(...
    feat_data_input, label_data_input, data_weight_input, one_model_info)


label_data=label_data_input;
feat_data=feat_data_input;
data_weight=data_weight_input;

feat_data.e_sel=one_model_info.e_sel;

if isempty(feat_data.e_sel)
    feat_data.sel_e_num=length(data_weight);
else
    feat_data.sel_e_num=nnz(feat_data.e_sel);
end


end



function one_model_info=gen_wl_model_info(wl_idx)


one_model_info=[];
one_model_info.sel_e_t=0;
one_model_info.sel_feat_t=0;
one_model_info.gen_wl_t=0;
one_model_info.apply_wl_t=0;
one_model_info.feat_mining_t=0;

one_model_info.tree_sel_feat_idxes=[];
one_model_info.sel_feat_idxes=[];

one_model_info.e_sel=[];
one_model_info.wl_idx=wl_idx;


end




