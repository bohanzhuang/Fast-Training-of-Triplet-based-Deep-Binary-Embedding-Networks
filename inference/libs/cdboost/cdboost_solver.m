
function [solver_result work_info]=cdboost_solver(train_info, work_info)



[solver_result work_info]=cdboost_solver_fast(train_info, work_info);
solver_result.handle_wl_remove_fn=@handle_wl_remove;


end






function [solver_result work_info]=cdboost_solver_fast(train_info, work_info)


tradeoff_nv=train_info.tradeoff_nv;


% use_mex=true;
use_mex=false;


train_cache=[];
if isfield(work_info, 'train_cache_sub')
    train_cache=work_info.train_cache_sub;
end

if isempty(train_cache)
       
    
    s_rescale_vs=work_info.pair_label_losses;
    
    wl_num=length(work_info.solver_wl_valid_sel);
    pair_num=size(s_rescale_vs,1);
    
    if use_mex
        train_cache.p_sel_mat=zeros(pair_num, wl_num ,'int8');
        train_cache.n_sel_mat=zeros(pair_num, wl_num ,'int8');
    else
        train_cache.p_sel_mat=false(pair_num, wl_num);
        train_cache.n_sel_mat=false(pair_num, wl_num);
    end
    
    train_cache.w=zeros(wl_num,1);
    train_cache.exp_m_wh=s_rescale_vs;
       
end

train_cache.use_mex=use_mex;


train_cache=update_cache_feat_change(work_info, train_cache);
update_dim_idxes=work_info.solver_update_wl_idxes;
assert(~isempty(update_dim_idxes))
    
use_stagewise=train_info.use_stagewise;
if use_stagewise
    tradeoff_nv=0;
else
    max_ws_iter=train_info.max_ws_iter;
    if max_ws_iter>1
        valid_dim_idxes=find(work_info.solver_wl_valid_sel);
        for ws_idx=2:max_ws_iter
            valid_dim_idxes=valid_dim_idxes(randperm(length(valid_dim_idxes)));
            update_dim_idxes=cat(1, update_dim_idxes, valid_dim_idxes);
        end
    end
end



init_w=train_cache.w;
exp_m_wh=train_cache.exp_m_wh;
    
failed=false;


    if use_mex
        
        % currently not support        
                        
    else
        
        [w exp_m_wh failed]=do_solve_stage(tradeoff_nv, update_dim_idxes, init_w,...
            train_cache.p_sel_mat, train_cache.n_sel_mat, exp_m_wh);
        
        wl_data_weight=exp_m_wh;
        obj_value=NaN;
        method_name='ada_stage';
        

       
    end
    
    



iter_num=length(update_dim_idxes);


train_cache.w=w;
train_cache.exp_m_wh=exp_m_wh;
work_info.train_cache_sub=train_cache;


solver_result.method=method_name;
solver_result.w=w;
solver_result.iter_num=iter_num;
solver_result.obj_value=obj_value;
solver_result.wlearner_pair_weight=wl_data_weight;
solver_result.failed=failed;

end




function [w exp_m_wh failed wl_flip]=do_solve(tradeoff_nv, update_dim_idxes, init_w, p_sel_mat, n_sel_mat, exp_m_wh)


    tmp_v1=tradeoff_nv/2;
    tmp_v2=tmp_v1^2;


    w=init_w;
    pair_num=size(p_sel_mat, 1);
            
    for up_idx=1:length(update_dim_idxes)
        
        up_dim_idx=update_dim_idxes(up_idx);
        p_sel=p_sel_mat(:, up_dim_idx);
        n_sel=n_sel_mat(:, up_dim_idx);

        one_old_w=w(up_dim_idx);
        if one_old_w>0
           tmp_m_wh=repmat(one_old_w, pair_num, 1); 
           tmp_m_wh(n_sel)=-tmp_m_wh(n_sel);
           exp_m_wh=exp_m_wh.*exp(tmp_m_wh); 
        end

        V_p=sum(exp_m_wh(p_sel));
        
        
%         error_num=nnz(n_sel);
        
        V_m=sum(exp_m_wh(n_sel));
        if V_m>eps
            one_new_w=0;
            if V_p>V_m && V_m>eps
               one_new_w=log(sqrt(V_p*V_m+tmp_v2) - tmp_v1) - log(V_m);
            end
        else
            one_new_w=log(V_p)+log(1e8);
        end
        

        w(up_dim_idx)=one_new_w;

        tmp_m_wh=repmat(one_new_w, pair_num, 1); 
        tmp_m_wh(p_sel)=-tmp_m_wh(p_sel);
        exp_m_wh=exp_m_wh.*exp(tmp_m_wh);

    end
    
    failed=false;
    if one_new_w<=0
        failed=true;
    end
    
    % if failed
    %     fprintf('\n\n------------cdboost: solver failed: error_num:%d, V_p:%.2e, V_m:%.2e\n\n', error_num, V_p, V_m);
    % end

end




function [w exp_m_wh failed]=do_solve_stage(tradeoff_nv, update_dim_idxes, init_w, p_sel_mat, n_sel_mat, exp_m_wh)

    w=init_w;
    pair_num=size(p_sel_mat, 1);
            
    up_dim_idx=update_dim_idxes;
    p_sel=p_sel_mat(:, up_dim_idx);
    n_sel=n_sel_mat(:, up_dim_idx);

    V_p=sum(exp_m_wh(p_sel));

    
    one_old_w=w(up_dim_idx);
    assert( one_old_w==0);
    
    
%     error_num=nnz(n_sel);
    V_m=sum(exp_m_wh(n_sel));
    if V_m>eps
        one_new_w=0;
        if V_p>V_m
            one_new_w=0.5*log(V_p/V_m);
        end
    else
        one_new_w=log(V_p)+log(1e8);
    end


    w(up_dim_idx)=one_new_w;

    tmp_m_wh=repmat(one_new_w, pair_num, 1); 
    tmp_m_wh(p_sel)=-tmp_m_wh(p_sel);
    exp_m_wh=exp_m_wh.*exp(tmp_m_wh);

    
    failed=false;
    if one_new_w<=0
        failed=true;
    end
    
    
    % if failed
        % fprintf('\n\n------cdboost solver failed: error_num:%d, V_p:%.2e, V_m:%.2e\n\n', error_num, V_p, V_m);
    % end
    

end











function train_cache=update_cache_feat_change(work_info, train_cache)

use_mex=train_cache.use_mex;
new_w_dim_idxes=work_info.solver_feat_change_idxes;

if ~isempty(new_w_dim_idxes)
    train_cache=update_cache_remove_dim(train_cache, new_w_dim_idxes);

    new_p_sel_mat=work_info.pair_feat(:,new_w_dim_idxes)>0;
    new_n_sel_mat=~new_p_sel_mat;
    if use_mex
        train_cache.p_sel_mat(:, new_w_dim_idxes)=int8(new_p_sel_mat);
        train_cache.n_sel_mat(:, new_w_dim_idxes)=int8(new_n_sel_mat);
    else
        train_cache.p_sel_mat(:, new_w_dim_idxes)=new_p_sel_mat;
        train_cache.n_sel_mat(:, new_w_dim_idxes)=new_n_sel_mat;
    end
end

end




function [work_info wl_data_weight]=handle_wl_remove(work_info, new_w_dim_idxes)

assert(~isempty(new_w_dim_idxes));

train_cache=work_info.train_cache_sub;
train_cache=update_cache_remove_dim(train_cache, new_w_dim_idxes);
work_info.train_cache_sub=train_cache;

wl_data_weight=work_info.train_cache_sub.exp_m_wh;

end



function train_cache=update_cache_remove_dim(train_cache, new_w_dim_idxes)

exp_m_wh=train_cache.exp_m_wh;
init_w=train_cache.w;
for tmp_idx=1:length(new_w_dim_idxes)
    one_w_idx=new_w_dim_idxes(tmp_idx);
    one_old_w=init_w(one_w_idx);
    if one_old_w>0
       n_sel=logical(train_cache.n_sel_mat(:, one_w_idx));
       pair_num=length(n_sel);
       tmp_m_wh=repmat(one_old_w, pair_num, 1); 
       tmp_m_wh(n_sel)=-tmp_m_wh(n_sel);
       exp_m_wh=exp_m_wh.*exp(tmp_m_wh); 
       
       init_w(one_w_idx)=0;
    end
end

train_cache.w=init_w;
train_cache.exp_m_wh=exp_m_wh;


end








