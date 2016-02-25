function f = makeimagestack(m,wantnorm,addborder,csize,bordersize)

% function f = makeimagestack(m,wantnorm,addborder,csize,bordersize)
%
% <m> is a 3D matrix.  if more than 3D, we reshape to be 3D.
%   we automatically convert to double format for the purposes of this function.
% <wantnorm> (optional) is
%   0 means no normalization
%   [A B] means normalize and threshold values such that A and B map to 0 and 1.
%   X means normalize and threshold values such that X percentile
%     from lower and upper end map to 0 and 1.  if the X percentile
%     from the two ends are the same, then map everything to 0.
%   -1 means normalize to 0 and 1 using -max(abs(m(:))) and max(abs(m(:)))
%   -2 means normalize to 0 and 1 using 0 and max(m(:))
%   -3 means normalize to 0 and 1 using min(m(:)) and max(m(:))
%   default: 0.
% <addborder> (optional) is
%    0 means do not add border
%    1 means add border at the right and bottom of each image.
%      the border is assigned the maximum value.
%    2 means like 1 but remove the final borders at the right and bottom.
%   -1 means like 1 but assign the border the middle value instead of the max.
%   -2 means like 2 but assign the border the middle value instead of the max.
%    j means like 1 but assign the border a value of 0.
%  2*j means like 2 but assign the border a value of 0.
%  NaN means plot images into figure windows instead of returning a matrix.
%      each image is separated by one matrix element from surrounding images.
%      in this case, <wantnorm> should not be 0.
%    default: 1.
% <csize> (optional) is [X Y], a 2D matrix size according
%   to which we concatenate the images (row then column).
%   default is [], which means try to make as square as possible
%   (e.g. for 16 images, we would use [4 4]).
%   special case is -1 which means use [1 size(m,3)].
%   another special case is [A 0] or [0 A] in which case we
%   set 0 to be the minimum possible to fit all the images in.
% <bordersize> (optional) is number of pixels in the border in the case that
%   <addborder> is not NaN.  default: 1.
%
% if <addborder> is not NaN, then return a 3D matrix.  the first two dimensions 
% contain images concatenated together, with any extra slots getting filled 
% with the minimum value.  the third dimension contains additional sets of images
% (if necessary).
%
% if <addborder> is NaN, then make a separate figure window for each set of images.
% (actually, we create new figure windows only for sets after the first set.  so the
% we attempt to draw the first set in the current figure window.)  in each figure window,
% we plot individual images using imagesc scaled to the range [0,1].
% we return <f> as [].
%
% example:
% a = randn(10,10,12);
% imagesc(makeimagestack(a,-1));
% imagesc(makeimagestack(a,-1,NaN));

% input
if ~exist('wantnorm','var') || isempty(wantnorm)
  wantnorm = 0;
end
if ~exist('addborder','var') || isempty(addborder)
  addborder = 1;
end
if ~exist('csize','var') || isempty(csize)
  csize = [];
end
if ~exist('bordersize','var') || isempty(bordersize)
  bordersize = 1;
end

% calc
nrows = size(m,1);
ncols = size(m,2);

% make double if necessary
m = double(m);
wantnorm = double(wantnorm);

% make <m> 3D if necessary
m = reshape(m,size(m,1),size(m,2),[]);

% find range, normalize
if length(wantnorm)==2
  m = normalizerange(m,0,1,wantnorm(1),wantnorm(2));
  mn = 0;
  mx = 1;
elseif wantnorm==0
  mn = nanmin(m(:));
  mx = nanmax(m(:));
elseif wantnorm==-1
  m = normalizerange(m,0,1,-max(abs(m(:))),max(abs(m(:))));
  mn = 0;
  mx = 1;
elseif wantnorm==-2
  m = normalizerange(m,0,1,0,max(m(:)));
  mn = 0;
  mx = 1;
elseif wantnorm==-3
  m = normalizerange(m,0,1,min(m(:)),max(m(:)));
  mn = 0;
  mx = 1;
else
  rng = prctile(m(:),[wantnorm 100-wantnorm]);
  if rng(2)==rng(1)
    m = zeros(size(m));  % avoid error from normalizerange.m
  else
    m = normalizerange(m,0,1,rng(1),rng(2));
  end
  mn = 0;
  mx = 1;
end
md = (mn+mx)/2;

% number of images
numim = size(m,3);

% calculate csize if necessary
if isempty(csize)
  rows = floor(sqrt(numim));  % MAKE INTO FUNCTION?
  cols = ceil(numim/rows);
  csize = [rows cols];
elseif isequal(csize,-1)
  csize = [1 numim];
elseif csize(1)==0
  csize(1) = ceil(numim/csize(2));
elseif csize(2)==0
  csize(2) = ceil(numim/csize(1));
end

% calc
chunksize = prod(csize);
numchunks = ceil(numim/chunksize);

% convert to cell vector, add some extra matrices if necessary
m = splitmatrix(m,3);
m = [m repmat({repmat(mn,size(m{1}))},1,numchunks*chunksize-numim)];

% figure case
if isnan(addborder)

  for p=1:numchunks
    if p ~= 1
      drawnow; figure;
    end
    hold on;
    for q=1:chunksize
      xx = linspace(1+(ceil(q/csize(1))-1)*(ncols+1),ncols+(ceil(q/csize(1))-1)*(ncols+1),ncols);
      yy = linspace(1+(mod2(q,csize(1))-1)*(nrows+1),nrows+(mod2(q,csize(1))-1)*(nrows+1),nrows);
      imagesc(xx,yy,m{(p-1)*chunksize+q},[0 1]);
    end
    axis equal;
    set(gca,'YDir','reverse');
  end
  f = [];

% matrix case
else

  % add border?
  if imag(addborder) || addborder
    for p=1:length(m)
      m{p}(end+(1:bordersize),:) = choose(imag(addborder),0,choose(addborder > 0,mx,md));
      m{p}(:,end+(1:bordersize)) = choose(imag(addborder),0,choose(addborder > 0,mx,md));
    end
  end
  
  % combine images
  f = [];
  for p=1:numchunks
    temp = m((p-1)*chunksize + (1:chunksize));
    f = cat(3,f,cell2mat(reshape(temp,csize)));
  end
  
  % remove final?
  if abs(addborder)==2
    f(end-bordersize+1:end,:,:) = [];
    f(:,end-bordersize+1:end,:) = [];
  end

end
