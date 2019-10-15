% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function cost = calculate_cost(infer_bi_code, infer_info)

pairwise_coefficients_matrix = infer_info.pairwise_weights_matrix;

cost = infer_bi_code' * pairwise_coefficients_matrix * infer_bi_code;

end
