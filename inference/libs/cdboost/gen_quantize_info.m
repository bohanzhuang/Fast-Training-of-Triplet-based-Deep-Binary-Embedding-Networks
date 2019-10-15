

function quantize_info=gen_quantize_info(X0)


quantize_info=[];


if ~isa(X0,'uint8')

	  
     xMin = min(X0);
     xMax = max(X0);
  
  xMin=double(xMin);
  xMax=double(xMax);

  one_extra_v=(xMax-xMin)/1000;
  one_extra_v=max(one_extra_v, eps);
  
  xMin=xMin-one_extra_v;
  xMax=xMax+one_extra_v;


	quantize_info.xMin=xMin;
	quantize_info.xMax=xMax;

end


end


