function [ multipse_label ] = update_multi_pseudo( x, c )
%UPDATE_MULTI_PSEUDO Summary of this function goes here
%   Detailed explanation goes here
f = vl_nnsoftmax(x);
f = reshape(f,size(x,3),[]);
iden_num = size(f,1);
label = 1/iden_num:1/iden_num:1;
multipse_label_temp = repmat(label',1,size(f,2));
[~, idx] = sort(f,1);
multipse_label = multipse_label_temp(idx);

gan = (c~=0);
multipse_label(:,gan) = 0;
multipse_label = reshape(multipse_label,1,1,size(multipse_label,1),size(multipse_label,2));

end

