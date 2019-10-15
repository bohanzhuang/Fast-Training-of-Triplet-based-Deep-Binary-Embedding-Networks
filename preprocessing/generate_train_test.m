
train_save_path = '/home/bohan/Documents/code/cvpr/data/cifar_10/train';
test_save_path = '/home/bohan/Documents/code/cvpr/data/cifar_10/test';

train_img_path = '/home/bohan/Documents/data/cifar_10/images/train';
test_img_path = '/home/bohan/Documents/data/cifar_10/images/test';

% prepare the labels by yourself
load train_labels;
load test_labels;


 for e_idx = 1:10
      
     relevant_idxes = find(train_labels==(e_idx-1));
        
     for k = 1:length(relevant_idxes)
         
        save_img_path = fullfile(train_save_path, num2str(e_idx));
 
        if ~isdir(save_img_path)
            mkdir(save_img_path);
        end
        img_path = fullfile(save_img_path, sprintf('%04d.JPEG', relevant_idxes(k)));
        raw_img_path = fullfile(train_img_path, sprintf('%04d.JPEG', relevant_idxes(k)));
       img = im2uint8(imread(raw_img_path));
       img = imresize(img, [224, 224]);   
       imwrite(img, img_path, 'jpg', 'Quality', 100);
        
         
     end 
        
     
 end


for e_idx = 1:10
    
    relevant_idxes =find(test_labels == (e_idx-1));
    

    for j = 1:length(relevant_idxes)
        
       save_img_path = fullfile(test_save_path, num2str(e_idx));

       if ~isdir(save_img_path)
           mkdir(save_img_path);
       end
       img_path = fullfile(save_img_path, sprintf('%04d.JPEG', relevant_idxes(j)));
       raw_img_path = fullfile(test_img_path, sprintf('%04d.JPEG', relevant_idxes(j)));
       img = im2uint8(imread(raw_img_path));
       img = imresize(img, [224, 224]);   
       imwrite(img, img_path, 'jpg', 'Quality', 100);
       
        
    end
end
