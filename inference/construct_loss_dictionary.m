% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function loss_dictionary = construct_loss_dictionary(update_bit)


pre_dist_1 = linspace(0, update_bit-1, update_bit);
pre_dist_2 = pre_dist_1;
distance_map = zeros(update_bit, update_bit);
loss_dictionary = cell(update_bit, update_bit);

one_loss_vec = [0;1;-1;0;0;-1;1;0];

m = update_bit/2;


% generate distance 
for i = 1: length(pre_dist_1)
    distance_map(i,:) =  bsxfun(@minus, pre_dist_2, pre_dist_1(i));
    
end


for i = 1:update_bit
    for j = 1:update_bit
        loss_dictionary{i,j} = max(0, m-(distance_map(i,j) + one_loss_vec));       
    end
end


end
