function [f,gaus,indices,info,filters] = applymultiscalegaussianfilters(images,sds,spacings,thresh,scaling,mode)

% function [f,gaus,indices,info,filters] = applymultiscalegaussianfilters(images,sds,spacings,thresh,scaling,mode)
%
% <images> is images x pixels.  it is assumed that the images are square.
% <sds> is 
%   (1) a vector with standard deviations.  this is the Gaussian filter case.
%   (2) a cell vector with elements that are [A B C].  this is the difference-of-Gaussians (DoG)
%       filter case.  in this case, A is the standard deviation of the surround Gaussian,
%       B is the standard deviation of the center Gaussian (should be smaller than A),
%       and C is a number whose absolute value is in [0,1].  C determines the volume of the
%       surround Gaussian relative to the volume of the center Gaussian.  for example, if C
%       is 1, the volume of the surround Gaussian is the same as the volume of the center
%       Gaussian and so the resulting filters should be close to zero-mean.  (they may
%       not be exactly zero-mean because of truncation and limited pixel resolution.)  the sign 
%       of C determines the polarity of the filters.  positive C indicates that the center
%       Gaussian should be positive and the surround Gaussian should be negative, whereas
%       negative C indicates the opposite.
% <spacings> is a vector of the same length as <sds> with the number of std devs
%   of the Gaussian envelope that should separate adjacent Gaussians along each dimension.
%   we calculate this number in terms of pixel units and then round down (to be conservative).
%   can be a scalar, in which case we use this value for all <sds> cases.
%   special case is a cell vector of the same length as <sds>,
%     where each element is the indices vector to use for a given case
%     (following the position conventions of filter2.m.)
%   note that in the DoG case, <spacings> refers to the surround Gaussian.
% <thresh> is a value.  points in the Gaussian envelopes less than <thresh> are set to 0.
%   in the Gaussian case, the filters are already created such that the max theoretical
%     value is 1, so <thresh> is to be interpreted with respect to this.
%   in the DoG case, the DoG filter is first scaled such that the max abs value of the
%     filter is 1 and then <thresh> is applied after that.
% <scaling> is
%   [0 X] means put filter into range [0,X].  0 implicitly means [0 1].
%   1 means make each filter unit-length (L2-norm equal to 1)
%   2 means make each filter have L1-norm equal to 1
% <mode> (optional) is
%   0 means center the filters on the center of pixels
%   1 means center the filters on the edge between pixels
%   default: 0.
%
% compute dot-product of <images> with a set of Gaussian or Difference-of-Gaussians filters.
% return <f> as images x channels.  the channels are ordered according to:
%   [position (down then right)][sds].  the precision of <f> will be the 
%   same as the precision of <images> (e.g. single).
% also, return the filters that we used in <gaus>, a cell matrix of
%   dimensions length(<sds>) x 1.  note that we crop the
%   filters, so the size may differ across scales.
% also, return the filters in <filters>, a sparse matrix of dimensions pixels x channels.  do not
%   assign this output unless you actually need it since in some cases we do not need to compute it.
% also, return the tiling indices in <indices>, a cell vector of dimensions 1 x length(<sds>).
% also, return information about the filters in <info>, a matrix of dimensions 4 x channels.
%   the entries are x, y, resolution, sd index
%
% Filters are tiled according to <spacings>.  at each scale, we fit as many filters
% as we can along each dimension (where a filter "fits" if its center is within the 
% field-of-view, exclusive at both ends), and center the set of filters with respect
% to the image (if uneven (due to odd number of pixels), use less excess at the left/top).
% the exact same filter is used across different positions; the heavy lifting
% is performed by filter2subsample.m.
%
% note that the image is zero-padded to accommodate filters that extend beyond the image.
%
% example:
% im = getsampleimage;
% f = applymultiscalegaussianfilters(flatten(im),[10],2,.01,0);
% dim = sqrt(size(f,2));
% figure; imagesc(im); axis equal tight;
% figure; imagesc(reshape(f,[dim dim])); axis equal tight;
%
% *** see also applymultiscalegaborfilters.m ***
%
% history:
% 2012/01/22 - add DoG case.
% 2011/12/09 - add <mode> input.  also, we now make the default mode to be 0 
%              (center on pixels).  this changes previous behavior!

% constants
wantdebug = 0;

% calc
numsc = length(sds);

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if ~iscell(spacings) && isscalar(spacings)
  spacings = repmat(spacings,[1 numsc]);
end

% calc
res = sqrt(size(images,2)); assert(res==round(res));

% debug mode stuff
if wantdebug
  figure; hold on; axis equal;
end

% do it
f = cast([],class(images)); gaus = {}; filters = sparse([]); indices = {}; info = []; 
for p=1:numsc

  % construct the filter.  since the filter might extend outside the image, 
  % we might have to construct the filter at a higher image resolution.
  resfactor = 1;
  while 1

    % construct the filter
    if mode==0
      ttt = floor(res*resfactor/2)*2+1;
    else
      ttt = res*resfactor;
    end
    if iscell(sds)
  
      % make the DoG and then scale it.
      gau = makedog2d(ttt,[],[],sds{p}(2),sds{p}(1)/sds{p}(2),abs(sds{p}(3)));
      if sds{p}(3) < 0
        gau = -gau;
      end
      gau = gau / max(abs(gau(:)));
     
      % ok, let's figure out smallest circle such that outside the circle, all values
      % are below the threshold.  but do it from out to in to avoid the zero-crossing 
      % areas in the middle.  (it would be dumb to just find all pixels smaller than thresh.)
      badr = ttt/2*sqrt(2);  % start with circle that circumscribes the square
      xx = []; yy = [];
      [d,xx,yy] = makecircleimage(ttt,badr,xx,yy,[],0);  % precompute xx and yy for speed
      while 1
        bad = logical(1-makecircleimage(ttt,badr,xx,yy,[],0));  % 1s beyond some radius
        % if we aren't marking any pixels as bad OR if all of the marked pixels are below the threshold
        if ~any(bad(:)) || all(abs(flatten(gau(bad))) < thresh)
          badr = badr - 1;  % this might not be fine-grained enough, but who really cares
        else
          badr = badr + 1;  % go back to the last valid value
          break;
        end
      end
      bad = logical(1-makecircleimage(ttt,badr,xx,yy,[],0));  % this indicates all the bad pixels

    else

      % make the Gaussian
      gau = makegaussian2d(ttt,[],[],sds(p),sds(p));

      % find where the Gaussian mask falls below the threshold
      bad = gau < thresh;

    end
    
    % if the mask is valid for all rows, this is bad (since we probably didn't cover the whole filter)
    if all(~all(bad,2));
      resfactor = resfactor + 1;
    else
      break;
    end

  end

  % set portions of filter beyond the threshold to 0
  gau(bad) = 0;
  
  % crop filter
  gau = gau(~all(bad,2),~all(bad,1));
  gausize = size(gau,1);

  % scale filter
  switch scaling(1)
  case 0
    if length(scaling) > 1 && scaling(2)~=1
      gau = gau * scaling(2);
    end
  case 1
    % make L2 unit-length
    gau = unitlength(gau);
  case 2
    % make L1 unit-length
    gau = l1unitlength(gau);
  end
  
  % record for safe keeping
  gaus{p,1} = gau;
  
  % if not a cell, we have to figure out indices of the subsampling grid
  if ~iscell(spacings)
    if iscell(sds)
      sep = floor(spacings(p)*sds{p}(1) + eps);  % separated by this many pixels (the eps is a hack to avoid weird rounding issues)
    else
      sep = floor(spacings(p)*sds(p) + eps);  % separated by this many pixels (the eps is a hack to avoid weird rounding issues)
    end
    if mod(gausize,2)==0
      indices{p} = 1:sep:res-1;                                           % let's see how many we can fit (res-1 because the last output is an odd man out)
      indices{p} = floor(((res-1)-indices{p}(end))/2) + 1 : sep : res-1;  % evenly distribute excess (if odd, less excess at beginning)
    else
      indices{p} = 1:sep:res;                                             % let's see how many we can fit
      indices{p} = floor((res-indices{p}(end))/2) + 1 : sep : res;        % evenly distribute excess (if odd, less excess at beginning)
    end
  % if a cell, then we have the indices already
  else
    indices{p} = spacings{p};
  end
  
  % report useful diagnostics
  fprintf('grid is %d x %d, gausize is %d, indices is %s\n',length(indices{p}),length(indices{p}),gausize,mat2str(indices{p}));
  if wantdebug
    adjustment = choose(mod(gausize,2)==0,-(gausize/2-1),-(gausize-1)/2);  % see filter2subsample.m
    for c0=1:length(indices{p})
      for r0=1:length(indices{p})
        plotrectangle([indices{p}(r0)+adjustment-.5 indices{p}(r0)+adjustment+gausize-1+.5 ...
                       indices{p}(c0)+adjustment-.5 indices{p}(c0)+adjustment+gausize-1+.5],[getcolorchar(p) '-']);
      end
    end
  end

  % figure out info0
  info0 = [];
  info0(1,:) = flatten(repmat(indices{p} + choose(mod(gausize,2)==0,.5,0),[length(indices{p}) 1]));
  info0(2,:) = flatten(repmat(indices{p}' + choose(mod(gausize,2)==0,.5,0),[1 length(indices{p})]));
  info0(3,:) = res;
  info0(4,:) = p;

  % ok, calculate filter output
  if nargout==5
    [filteroutput,filters0] = filter2subsample(gau,permute(reshape(images,[],res,res),[2 3 1]),indices{p});
    filters = cat(2,squish(filters0,2),filters);  % NOTE: add to beginning
  else
    filteroutput = filter2subsample(gau,permute(reshape(images,[],res,res),[2 3 1]),indices{p});
  end
  filteroutput = squish(filteroutput,2)';  % images x n*n

  % take results and add them to our final results
  f = [f filteroutput];
  info = [info info0];

end
