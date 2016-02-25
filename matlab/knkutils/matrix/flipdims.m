function m = flipdims(m,dims)

% function m = flipdims(m,dims)
%
% <m> is a matrix
% <dims> is a vector of 0s and 1s indicating which dimensions to flip
%
% flip <m> according to <dims>.
%
% example:
% flipdims([1 2; 3 4],[0 1])

for p=1:length(dims)
  if dims(p)
    m = flipdim(m,p);
  end
end
