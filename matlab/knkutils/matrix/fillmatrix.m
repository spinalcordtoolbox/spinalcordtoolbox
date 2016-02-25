function f = fillmatrix(v,msize,dim)

% function f = fillmatrix(v,msize,dim)
%
% <v> is a vector
% <msize> is a matrix size
% <dim> is the dimension along which <v> is oriented
%
% return a matrix of size <msize> filled with <v>.
%
% example:
% isequal(fillmatrix([1 2 3],[3 2],1),[1 1; 2 2; 3 3])

% calc
len = length(msize);
temp = ones(1,max(len,dim));

% first, reshape
temp1 = temp;
if dim <= len
  temp1(dim) = msize(dim);
end
f = reshape(v,temp1);

% then, repmat
temp2 = temp;
temp2(1:len) = msize;
temp2(dim) = 1;
f = repmat(f,temp2);
