# Paper: "Fast Training of Triplet-based Deep Binary Embedding Networks"

##People
Bohan Zhuang, Guosheng Lin, Chunhua Shen and Ian Reid.
Code author: Bohan Zhuang
This code is provided for non-profit research purpose only; and is released under the GNU license. 
For commercial applications, please contact Chunhua Shen http://www.cs.adelaide.edu.au/~chhshen/.

__This is the implementation of the following paper. If you use this code in your research, please cite our paper__

```
@InProceedings{Zhuang_2016_CVPR,
author = {Zhuang, Bohan and Lin, Guosheng and Shen, Chunhua and Reid, Ian},
title = {Fast Training of Triplet-Based Deep Binary Embedding Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2016}
}


```

## Overview
./step1/ includes the code for the binary codes inference step.
./lib/ includes the necessary codes for the network training. We inplement it using Theano. 
./preprocessing is the data preprocessing toolbox. 

## Data Preprocessing
The data preprocessing code is in ./preprocessing/ and it will generate suitable data for training and testing. Please modify ./preprocessing/paths.yaml.

run make_caffe_txt.py-->make_hkl.py-->make_labels.py 


The processed data is in folder ./preprocessed_data/.


## Training

The code is based on Ubuntu 14.04.
The main function is the ./step1/train.m file.
Please modify the configurations in the config.yaml.
Trained models will be stored in ./models/. 

## Testing

You should first extract the gallery codes as well as the query codes by running ./code_extraction.py.
Then run ./test.m to do testing. 


## Copyright

Copyright (c) Bohan Zhuang. 2016.

** This code is for non-commercial purposes only. For commerical purposes,
please contact Chunhua Shen <chhshen@gmail.com> **

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
