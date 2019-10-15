
% modified by Guosheng Lin
% contact: guosheng.lin@gmail.com
% originally from Pitor Dollar's Toolbox.


function tree = binaryTreeTrain_sample( data, varargin )
% Train binary decision tree classifier.
%
% Highly optimized code for training decision trees over binary variables.
% Training a decision stump (depth=1) over 5000 features and 10000 training
% examples takes 70ms on a single core machine and *7ms* with 12 cores and
% OpenMP enabled (OpenMP is enabled by default, see toolboxCompile). This
% code shares similarities with forestTrain.m but is optimized for binary
% labels. Moreover, while forestTrain is meant for training random decision
% forests, this code is tuned for use with boosting (see adaBoostTrain.m).
%
% For more information on how to quickly boost decision trees see:
%   [1] R. Appel, T. Fuchs, P. Doll�r, P. Perona; "Quickly Boosting
%   Decision Trees � Pruning Underachieving Features Early," ICML 2013.
% The code here implements a simple brute-force strategy with the option to
% sample features used for training each node for additional speedups.
% Further gains using the ideas from the ICML paper are possible. If you
% use this code please consider citing our ICML paper.
%
% During training each feature is quantized to lie between [0,nBins-1],
% where nBins<=256. Quantization is expensive and should be performed just
% once if training multiple trees. Note that the second output of the
% algorithm is the quantized data, this can be reused in future training.
%
% USAGE
%  [tree,data,err] = binaryTreeTrain( data, [pTree] )
%
% INPUTS
%  data       - data for training tree
%   .X0         - [N0xF] negative feature vectors
%   .X1         - [N1xF] positive feature vectors
%   .wts0       - [N0x1] negative weights
%   .wts1       - [N1x1] positive weights
%   .xMin       - [1xF] optional vals defining feature quantization
%   .xStep      - [1xF] optional vals defining feature quantization
%   .xType      - [] optional original data type for features
%  pTree      - additional params (struct or name/value pairs)
%   .nBins      - [256] maximum number of quanizaton bins (<=256)
%   .maxDepth   - [1] maximum depth of tree
%   .minWeight  - [.01] minimum sample weigth to allow split
%   .fracFtrs   - [1] fraction of features to sample for each node split
%   .nThreads   - [inf] max number of computational threads to use
%
% OUTPUTS
%  tree       - learned decision tree model struct w the following fields
%   .fids       - [Kx1] feature ids for each node
%   .thrs       - [Kx1] threshold corresponding to each fid
%   .child      - [Kx1] index of child for each node (1-indexed)
%   .hs         - [Kx1] log ratio (.5*log(p/(1-p)) at each node
%   .weights    - [Kx1] total sample weight at each node
%   .depth      - [Kx1] depth of each node
%  data       - data used for training tree (quantized version of input)
%  err        - decision tree training error
%
% EXAMPLE
%
% See also binaryTreeApply, adaBoostTrain, forestTrain
%
% Piotr's Image&Video Toolbox      Version 3.21
% Copyright 2013 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see external/bsd.txt]

t0=tic;

default_min_node_size=5;

% turn this off...
default_purity_thresh=0;


% get parameters
dfs={'nBins',256,'maxDepth',1,'minWeight',.01,'fracFtrs',1,'nThreads',8, 'max_e_num', inf,...
    'e_sel_X0', [], 'e_sel_X1', [], 'min_node_size', default_min_node_size, 'purity_thresh', default_purity_thresh};

[nBins,maxDepth,minWeight,fracFtrs, nThreads, max_e_num, e_sel_X0, e_sel_X1,...
    min_node_size, purity_thresh]=getPrmDflt(varargin,dfs,1);

assert(nBins<=256);

% get data and normalize weights
dfs={ 'X0','REQ', 'X1','REQ', 'wts0',[], 'wts1',[], ...
  'xMin',[], 'xStep',[], 'xType',[] };
[X0,X1,wts0,wts1,xMin,xStep,xType]=getPrmDflt(data,dfs,1);


wts0=single(wts0);
wts1=single(wts1);

[N0,F]=size(X0); [N1,F1]=size(X1); assert(F==F1);


% try
assert( ~isempty(e_sel_X0));
assert( ~isempty(e_sel_X1));
% catch err_info
%     dbstack;
%     keyboard;
% end

wts0=wts0(e_sel_X0);
N0=length(wts0);
e_sel_idxes0_cpp=uint32(find(e_sel_X0)-1);

wts1=wts1(e_sel_X1);
N1=length(wts1);
e_sel_idxes1_cpp=uint32(find(e_sel_X1)-1);




%TODO: why this cannot have the same result witht the original one?????



if(isempty(xType)), xMin=zeros(1,F); xStep=ones(1,F); xType=class(X0); end
assert(isfloat(wts0)); if(isempty(wts0)), wts0=ones(N0,1)/N0; end
assert(isfloat(wts1)); if(isempty(wts1)), wts1=ones(N1,1)/N1; end


% if dataset is very large, this normalization will have very small value..
% seems not necessary:
% w=sum(wts0)+sum(wts1); if(abs(w-1)>1e-3), wts0=wts0/w; wts1=wts1/w; end




% train decision tree classifier
% K=2*(N0+N1);
K=min(2*(N0+N1), 2^(maxDepth+1));

thrs=zeros(K,1,xType);
hs=zeros(K,1,'single'); weights=hs; errs=hs;
fids=zeros(K,1,'uint32'); child=fids; depth=fids;


wtsAll0=cell(K,1); wtsAll0{1}=wts0;
wtsAll1=cell(K,1); wtsAll1{1}=wts1; 

e_sel_All0=cell(K,1); e_sel_All0{1}=e_sel_idxes0_cpp;
e_sel_All1=cell(K,1); e_sel_All1{1}=e_sel_idxes1_cpp; 


valid_sel_All0=cell(K,1); valid_sel_All0{1}=[];
valid_sel_All1=cell(K,1); valid_sel_All1{1}=[]; 



k=1; K=2;



fidsSt=1:F;

other_time=0;
% other_time=other_time+toc(t0);

e_sel_time=0;
train_time=0;




while( k < K )
    
  tmp_t=tic;
    
    % get node weights and prior
      wts0=wtsAll0{k}; wtsAll0{k}=[]; 
      wts1=wtsAll1{k}; wtsAll1{k}=[]; 

      %here need a normalization step, to prevent the child node predict the same label, 
    %if the example weighting is not balanced, TO verify!!!!!!!!!!
    % but this will cause the acc_rate>0.5 error, why????????
%       parent_sum_wts0=sum(wts0);
%       parent_sum_wts1=sum(wts1);


      e_sel_idxes0_cpp=e_sel_All0{k}; e_sel_All0{k}=[]; 
      e_sel_idxes1_cpp=e_sel_All1{k}; e_sel_All1{k}=[]; 
            
      tmp_sel0=valid_sel_All0{k}; valid_sel_All0{k}=[]; 
      tmp_sel1=valid_sel_All1{k}; valid_sel_All1{k}=[]; 
      
%           tmp_sel0=wts0>eps;
%           tmp_sel1=wts1>eps;
      
      if ~isempty(tmp_sel0)
          wts0=wts0(tmp_sel0);
          e_sel_idxes0_cpp=e_sel_idxes0_cpp(tmp_sel0);
      end
      
      if ~isempty(tmp_sel1)
          wts1=wts1(tmp_sel1);
          e_sel_idxes1_cpp=e_sel_idxes1_cpp(tmp_sel1);
      end
      

      w0=sum(wts0);
      w1=sum(wts1);


      w=w0+w1; 
      weights(k)=w; 

      
      prior=w1/w; 
      

      errs(k)=min(prior,1-prior);
      hs(k)=max(-4,min(4,.5*log(prior/(1-prior))));
      
      
      

      node_stop=false;
      if depth(k)>=maxDepth 
            node_stop=true;
      end
      

      if  length(wts0) <= min_node_size || length(wts1) <= min_node_size
          node_stop=true;
      end
      
      
      if node_stop
          k=k+1; continue; 
      end
     
      
      % train best stump
      if(fracFtrs<1), fidsSt=randsample(F,floor(F*fracFtrs)); end
  
  other_time=other_time+toc(tmp_t);
     
      
      tmp_t=tic;
           
      [tmp_wts0, tmp_e_sel_idxes0_cpp, tmp_w0]=gen_one_selection(wts0, e_sel_idxes0_cpp, w0, max_e_num);
      [tmp_wts1, tmp_e_sel_idxes1_cpp, tmp_w1]=gen_one_selection(wts1, e_sel_idxes1_cpp, w1, max_e_num);

      tmp_w=tmp_w0+tmp_w1;
      tmp_wts0=tmp_wts0/tmp_w;
      tmp_wts1=tmp_wts1/tmp_w;
      tmp_prior=tmp_w1/tmp_w;
      
      tmp_trn_t=tic;
      
      [errsSt,thrsSt] = binaryTreeTrain1(X0,X1,single(tmp_wts0),...
            single(tmp_wts1),nBins,tmp_prior,uint32(fidsSt-1),nThreads, tmp_e_sel_idxes0_cpp, tmp_e_sel_idxes1_cpp);
        

      one_train_time=toc(tmp_trn_t);
      train_time=train_time+one_train_time;
     
        
      e_sel_time=e_sel_time+toc(tmp_t)-one_train_time;
      
  
      tmp_t=tic;
      


          [~,fid]=min(errsSt); thr=single(thrsSt(fid))+.5; fid=fidsSt(fid);

          e_sel_idxes0=e_sel_idxes0_cpp+1;
          e_sel_idxes1=e_sel_idxes1_cpp+1;
          
          left0=X0(e_sel_idxes0,fid)<thr;   
          left1=X1(e_sel_idxes1,fid)<thr;


      if( (any(left0)||any(left1)) && (any(~left0)||any(~left1)) )
        thr = xMin(fid)+xStep(fid)*thr;
        child(k)=K; fids(k)=fid-1; thrs(k)=thr;

		
        wtsAll0{K}=wts0; wtsAll0{K+1}=wts0;
        wtsAll1{K}=wts1; wtsAll1{K+1}=wts1;
        
        valid_sel_All0{K}=left0;
        valid_sel_All1{K}=left1;
        valid_sel_All0{K+1}=~left0;
        valid_sel_All1{K+1}=~left1;


        e_sel_All0{K}=e_sel_idxes0_cpp;
        e_sel_All1{K}=e_sel_idxes1_cpp;
        e_sel_All0{K+1}=e_sel_idxes0_cpp;
        e_sel_All1{K+1}=e_sel_idxes1_cpp;


        depth(K:K+1)=depth(k)+1; 
        
        K=K+2;
    
        other_time=other_time+toc(tmp_t);
    
      end 
       
       k=k+1;
       
end 


K=K-1;



% create output model struct
tree=struct('fids',fids(1:K),'thrs',thrs(1:K),'child',child(1:K),...
  'hs',hs(1:K),'weights',weights(1:K),'depth',depth(1:K));



tree.e_sel_time=e_sel_time;
tree.train_time=train_time;
tree.other_time=other_time;
tree.total_time=toc(t0);





end


function [tmp_wts0, tmp_e_sel_idxes0_cpp, tmp_w0]=gen_one_selection(wts0, e_sel_idxes0_cpp, w0, max_e_num)

      
      tmp_wts0=wts0;
      tmp_e_sel_idxes0_cpp=e_sel_idxes0_cpp;
      tmp_w0=w0;
      
      
      N0_tmp=length(tmp_wts0);
      if max_e_num<N0_tmp
          % weighted sampling:
        
          tmp_sel_idxes=randsample(N0_tmp, max_e_num);
                  
          tmp_e_sel_idxes0_cpp=tmp_e_sel_idxes0_cpp(tmp_sel_idxes);
          tmp_wts0=tmp_wts0(tmp_sel_idxes);
          
          tmp_w0=sum(tmp_wts0);
      end
      

end


