% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function infer_block_info = gen_block_multiclass(train_info)

  train_label_map = train_info.train_affinity_map;
 g_num = size(train_label_map, 2);
 infer_block_info = cell(g_num,1);
 for  g_idx = 1:g_num
     
     one_group_idxes = find(train_label_map(:,g_idx));
    if ~isempty(one_group_idxes)
        infer_block_info{g_idx}=one_group_idxes;
    end
     
 end


end
