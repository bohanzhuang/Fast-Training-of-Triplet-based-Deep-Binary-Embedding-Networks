

function q_X0=quantize_data(X0, quantize_info)


nBins=256;

if islogical(X0)
	q_X0=uint8(full(X0));
end

if ~isa(X0,'uint8')

	if issparse(X0)
		X0=double(X0);
	else
		X0=single(X0);
	end

  xMin = quantize_info.xMin;
  xMax = quantize_info.xMax;
  xStep = (xMax-xMin) / (nBins-1);


  work_step=ceil(1e4*1e3/length(xStep));
  e_counter=0;
  e_num=size(X0, 1);

  q_X0=zeros(size(X0), 'uint8');

  	while e_counter<e_num

  		end_idx=e_counter+work_step;
  		end_idx=min(end_idx, e_num);
  		sel_e_idxes=e_counter+1:end_idx;
  		e_counter=end_idx;

 		one_X0 = bsxfun(@times,bsxfun(@minus,X0(sel_e_idxes,:),xMin),1./xStep);
        one_X0=uint8(full(one_X0));

 		q_X0(sel_e_idxes,:)=one_X0;
	end

end



end