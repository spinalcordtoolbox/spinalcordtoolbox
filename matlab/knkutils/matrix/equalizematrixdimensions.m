function [m1,m2] = equalizematrixdimensions(m1,m2)

% function [m1,m2] = equalizematrixdimensions(m1,m2)
%
% <m1> is a matrix (cell matrix okay)
% <m2> is a matrix (cell matrix okay)
%
% make <m1> and <m2> the same dimensions by expanding these matrices
% as necessary and filling any new entries with NaN or [].
% <m1> and <m2> must be of the same type (matrix or cell matrix).
%
% example:
% [f,g] = equalizematrixdimensions([1 2],[1 3]');
% isequalwithequalnans(f,[1 2; NaN NaN])
% isequalwithequalnans(g,[1 NaN; 3 NaN])

% calc
dim = max(ndims(m1),ndims(m2));

% figure out the new dimensions
newdim = zeros(1,dim);
for p=1:dim
  newdim(p) = max(size(m1,p),size(m2,p));
end

% do it
if iscell(m1)
  m1 = placematrix2(cell(newdim),m1);
  m2 = placematrix2(cell(newdim),m2);
else
  m1 = placematrix2(repmat(NaN,newdim),m1);
  m2 = placematrix2(repmat(NaN,newdim),m2);
end
