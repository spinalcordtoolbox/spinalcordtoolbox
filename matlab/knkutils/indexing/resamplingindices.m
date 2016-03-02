function f = resamplingindices(x,y,n)

% function f = resamplingindices(x,y,n)
%
% <x> is the beginning index (integer)
% <y> is the ending index (integer).  <y> must be >= <x>.
% <n> is
%   A means the number of indices desired (positive integer)
%  -B means upsample by a factor of B (positive number)
%
% treat <x> and <y> as matrix indices.  return a vector of 
% indices that correspond to resampling according to <n>.
% the returned indices may be non-integral.  the field-of-view
% spanned by <x> through <y> is preserved when determining
% the new indices.  for example, <x>==2 and <y>==5 means
% that the total field-of-view proceeds from 1.5 to 5.5.
%
% example:
% x = resamplingindices(1,10,20);
% figure; hold on;
% scatter(1:10,ones(1,10),'r.');
% scatter(x,2*ones(1,20),'b.');
% axis([0 11 0 3]);

if n > 0
  f = linspacefixeddiff(x-.5 + (y-x+1)/n/2,(y-x+1)/n,n);
else
  f = linspacefixeddiff(x-.5 + 1/-n/2,1/-n,(y-x+1)*-n);
end
