
function bi_code=gen_bi_code_from_logical(hs_feat)

assert(islogical(hs_feat));

hs_feat=full(hs_feat);
bi_code=-ones(size(hs_feat), 'int8');
bi_code(hs_feat)=1;    

end

