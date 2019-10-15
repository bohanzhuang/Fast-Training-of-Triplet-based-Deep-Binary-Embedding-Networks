% Code author: Bohan Zhuang. 
%Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com

function pairwise_coefficients_dictionary = construct_coefficients_dictionary(loss_dictionary)

             
scalar_matrix = [1,1,1,1,1,1,1,1,1;
                 1,1,-1,1,1,-1,-1,-1,1;
                 1,-1,1,-1,1,-1,1,-1,1;
                 1,-1,-1,-1,1,1,-1,1,1;
                 1,-1,-1,-1,1,1,-1,1,1;
                 1,-1,1,-1,1,-1,1,-1,1;
                 1,1,-1,1,1,-1,-1,-1,1;
                 1,1,1,1,1,1,1,1,1];
                 


 size_1 = size(loss_dictionary, 1);
 size_2 = size(loss_dictionary, 2);
 
 pairwise_idxes = [4,7,2,8,3,6];
 
 
 pairwise_coefficients_dictionary = cell(size_1, size_2);
 
 for i = 1:size_1
     for j = 1:size_2
         linear_coefficients = scalar_matrix \ loss_dictionary{i,j};
         pairwise_coefficients = linear_coefficients(pairwise_idxes); 
         pairwise_coefficients_dictionary{i,j} = pairwise_coefficients;           
         
     end
 end
 

end
