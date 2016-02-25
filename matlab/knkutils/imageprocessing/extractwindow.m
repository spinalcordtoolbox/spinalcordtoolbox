function f = extractwindow(im,windowsize,windowgrid,idx)

% function f = extractwindow(im,windowsize,windowgrid,idx)
%
% <im> is A x B, a large matrix
% <windowsize> is [C D] where C is window size in first dimension and
%   D is window size in second dimension
% <windowgrid> is [E F] where E means E windows distributed along
%   first dimension and F windows distributed along second dimension
% <idx> (optional) is indices of the windows to pull out.
%   indices must be between 1 and prod(windowgrid).
%   default is 1:prod(windowgrid).
%
% return window(s) from <im> (multiple windows along third dimension).
% we position windows such that the first and last windows along each
% dimension are maximally spaced apart, and then we evenly distribute
% windows between those two positions.  a special case is when the number
% of windows along a dimension is 1; in this case, we center the window
% with respect to this dimension.
%
% example:
% isequal(extractwindow([1 2 3 4 5 6],[1 2],[1 3],2),[3 4])

% input
if ~exist('idx','var') || isempty(idx)
  idx = 1:prod(windowgrid);
end

% calc
imsize = size(im);

% figure out starting positions
if windowgrid(1)==1
  rowstart = round((1 + (imsize(1)-windowsize(1)+1))/2);
else
  rowstart = round(linspace(1,imsize(1)-windowsize(1)+1,windowgrid(1)));
end
if windowgrid(2)==1
  colstart = round((1 + (imsize(2)-windowsize(2)+1))/2);
else
  colstart = round(linspace(1,imsize(2)-windowsize(2)+1,windowgrid(2)));
end

% do it
f = zeros(windowsize(1),windowsize(2),length(idx));
for p=1:length(idx)

  % figure out upper-left corner of window to pull out
  r = rowstart(mod2(idx(p),length(rowstart)));
  c = colstart(ceil(idx(p)/length(rowstart)));
  
  % extract
  f(:,:,p) = im(r-1+(1:windowsize(1)),c-1+(1:windowsize(2)));

end
