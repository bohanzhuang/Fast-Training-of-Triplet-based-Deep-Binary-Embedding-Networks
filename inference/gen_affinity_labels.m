% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function triplet_samples_idxes = gen_affinity_labels(train_data)


disp('gen_affinity_labels...');

e_num=size(train_data.label_data, 1);
label_data=train_data.label_data;
assert(size(label_data, 1)==e_num);
assert(size(label_data, 2)==1);

maximum_sample_num = 500;



triplet_samples_idxes = [];

for e_idx = 1:e_num

    relevant_sel=label_data(e_idx)==label_data;
    irrelevant_sel=~relevant_sel;
    relevant_sel(e_idx)=false;

    relevant_idxes=find(relevant_sel);
    irrelevant_idxes=find(irrelevant_sel);

    [A, B] = meshgrid(relevant_idxes, irrelevant_idxes);

    total_triplet_samples_idxes = [A(:) B(:)];

    bias = repmat(e_idx, size(total_triplet_samples_idxes,1), 1);

    sub_triplet_samples_idxes = [bias, total_triplet_samples_idxes];


    sub_triplet_samples_idxes=sub_triplet_samples_idxes(randsample(size(sub_triplet_samples_idxes, 1), maximum_sample_num), :);
   

    triplet_samples_idxes = cat(1, triplet_samples_idxes, sub_triplet_samples_idxes);
    

end

end
