function f = constructfiltersubsample(res,x,indices)

% function f = constructfiltersubsample(res,x,indices)
%
% <res> is number of pixels along one side of the larger matrix
% <x> is a 2D square matrix with the filter
% <indices> is a vector of indices indicating where the filter should
%   be placed with respect to each dimension of the larger matrix.
%   note that indices can lie outside of the range [1,max].
%
% return the filters as res x res x N*N where N is length(indices).
% portions of the filter that lie outside the range of the larger matrix
% are just ignored.  note that for even-sized filters, we use the position
% conventions of filter2.m.
%
% there is a special case, namely that <x> can have stuff along the third dimension.
% in this case (i.e. size(x,3) > 1), the filters are returned with size
% res x res x stuff x N*N.
%
% example:
% figure; imagesc(makeimagestack(constructfiltersubsample(16,[1 2; 1 2],[4 9 12])));

% calc
xn = size(x,1);
n = length(indices);
dimthree = size(x,3);

% init
z = zeros(res,res,dimthree);

% calc
adjustment = choose(mod(xn,2)==0,-(xn/2-1),-(xn-1)/2);

% construct f
f = zeros(res,res,dimthree,n*n);
for c=1:n
  for r=1:n
    f(:,:,:,(c-1)*n+r) = placematrix(z,x,[indices(r) indices(c)]+adjustment);
  end
end

% reshape if necessary
if dimthree==1
  f = reshape(f,res,res,n*n);
end
