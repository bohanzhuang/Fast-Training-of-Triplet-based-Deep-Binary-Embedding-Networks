% Code author: Bohan Zhuang. 
% Contact: bohan.zhuang@adelaide.edu.au or zhuangbohan2013@gmail.com



function test()

addpath(genpath([pwd '/']));

load './datasets/test_labels.mat'
load './datasets/database_labels.mat'


database_label = double(database_labels);
test_label = double(test_labels);




% process the data


one_database_code = load('database_code.mat');
one_test_code = load('test_code.mat');

fasthash_db_data_code = one_database_code.database_code;
fasthash_tst_data_code = one_test_code.test_code;



%fasthash_db_data_code = database_code.database_code';
%fasthash_tst_data_code = test_code.test_code';

fasthash_db_data_code = fasthash_db_data_code > 0;
fasthash_tst_data_code = fasthash_tst_data_code >0;

bit_num = 64;



label_type='multiclass';

db_label_info.label_data=database_label;
db_label_info.label_type=label_type;

test_label_info.label_data=test_label;
test_label_info.label_type=label_type;

code_data_info=[];
code_data_info.db_label_info=db_label_info;
code_data_info.test_label_info=test_label_info;

eva_bit_step=round(min(8, bit_num/4));
eva_bits=eva_bit_step:eva_bit_step:bit_num;

eva_param=[];
eva_param.eva_top_knn_pk=60000;
eva_param.eva_bits=eva_bits;
eva_param.code_data_info=code_data_info;
eva_param.test_method = 'map';


fasthash_predict_result=[];
fasthash_predict_result.method_name='FastHash';
fasthash_predict_result.eva_bits=eva_bits;

predict_results=cell(0);

for b_idx=1:length(eva_bits)
      one_bit_num=eva_bits(b_idx);
      fasthash_code_data_info=code_data_info;
      fasthash_code_data_info.db_data_code=fasthash_db_data_code(:, 1:one_bit_num);
      fasthash_code_data_info.tst_data_code=fasthash_tst_data_code(:, 1:one_bit_num);
      one_bit_result=hash_evaluate(eva_param, fasthash_code_data_info);
      fasthash_predict_result.map60000_eva_bits(b_idx)=one_bit_result.map60000;
end


predict_results{end+1}=fasthash_predict_result;
%-----------------------------------------------
% plot results
ds_name = 'NUS_WIDE';

f1=figure;
line_width=2;
xy_font_size=22;
marker_size=10;
legend_font_size=15;
xy_v_font_size=15;
title_font_size=xy_font_size;

legend_strs=cell(length(predict_results), 1);

for p_idx=1:length(predict_results)
    
    predict_result=predict_results{p_idx};
   
    fprintf('\n\n-------------predict_result--------------------------------------------------------\n\n');
    disp(predict_result);

    color=gen_color(p_idx);
    marker=gen_marker(p_idx);
    
    x_values=eva_bits;
    y_values=predict_result.map60000_eva_bits;

    p=plot(x_values, y_values);
    
    set(p,'Color', color)
    set(p,'Marker',marker);
    set(p,'LineWidth',line_width);
    set(p,'MarkerSize',marker_size);
    
    legend_strs{p_idx}=[predict_result.method_name  sprintf('(%.3f)', y_values(end))];
    
    hold all
end


hleg=legend(legend_strs);
h1=xlabel('Number of bits');
h2=ylabel('Precision @ K (K=100)');
title(ds_name, 'FontSize', title_font_size);

set(gca,'XTick',x_values);
xlim([x_values(1) x_values(end)]);
set(hleg, 'FontSize',legend_font_size);
set(hleg,'Location','SouthEast');
set(h1, 'FontSize',xy_font_size);
set(h2, 'FontSize',xy_font_size);
set(gca, 'FontSize',xy_v_font_size);
set(hleg, 'FontSize',legend_font_size);
grid on;
hold off



end


