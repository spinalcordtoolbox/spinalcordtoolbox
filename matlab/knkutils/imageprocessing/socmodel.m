function [response,cache] = socmodel(stimulus,res,rng,efactor,gparams,r,s,sd,spacing,n,c,cache)

% function [response,cache] = socmodel(stimulus)
%
% this will use defaults for all parameters.
%
%   OR
%
% function [response,cache] = socmodel(stimulus,res,rng,efactor,gparams,r,s,sd,spacing,n,c,cache)
%
% <stimulus> is a set of images.  <stimulus> should be A x A x N where A is the number 
%   of pixels along each dimension and N is the number of distinct images.
% <res> (optional) is the number of pixels to resize <stimulus> to (in order to save 
%   computational time).  default is [], which means to use 4 times the number of
%   cycles per field-of-view specified by the first element of <gparams>.  (we round
%   if necessary.)
% <rng> (optional) is [MIN MAX MEAN] where MIN is the lowest luminance value in 
%   <stimulus>, MAX is the highest luminance value in <stimulus>, and MEAN is the
%   "mean" luminance value in <stimulus>.  these values are used to apply a simple
%   linear rescaling and offsetting to <stimulus> --- specifically, we transform
%   <stimulus> such that MIN maps to 0 and MAX maps to 1 (outlier values are 
%   clipped) and then we subtract off a constant from <stimulus> such that
%   MEAN maps to 0.  for example, if the values in <stimulus> range between
%   0 and 254 with the mean-luminance at 127, then we could set <rng> to 
%   [0 254 127] and this will result in the values in <stimulus> being mapped to
%   the range [-0.5,0.5] with the mean-luminance mapped to 0.  default is to use
%   the empirically observed minimum, maximum, and mean of the values in <stimulus>.
% <efactor> (optional) is a number greater than or equal to 1 indicating how much 
%   enlargement of each dimension to perform (in order to minimize edge effects).
%   we round to the nearest whole number of pixels that results in identical padding
%   on all sides of the stimulus.  for example, <efactor> being 1.2 means to enlarge 
%   each stimulus dimension by 20%.  default: 1.2.
% <gparams> (optional) is a cell vector with arguments to applymultiscalegaborfilters.m,
%   ignoring the first input 'images' and starting with the second input 'cpfovs'.  
%   these arguments determine the design of the Gabor filters.  the first entry in <gparams>
%   is the 'cpfovs' input and must be a scalar indicating a single number of cycles per
%   field-of-view (it cannot be a vector).  also, the fifth entry in <gparams> is the
%   'numph' input and must be 2.  default: {CPFOV -1 1 8 2 .01 2 0} where CPFOV is A/4, 
%   that is, 1/4 times the resolution of <stimulus>.
% <r> (optional) is the exponent in the divisive-normalization operation.  default: 1.
% <s> (optional) is the semi-saturation constant in the divisive-normalization operation.
%   default: 0.5.
% <sd> (optional) is the standard deviation of the 2D Gaussian that is applied to the
%   contrast image.  note that the units are in pixels of the contrast image.  
%   default: 1.5.
% <spacing> (optional) is the number of standard deviations that should separate
%   adjacent 2D Gaussians.  default: 1/1.5.  (this default, combined with the default
%   for <sd>, makes it such that the grid of model outputs has the same resolution as
%   the contrast image.)
% <n> (optional) is the exponent of the power-law nonlinearity.  default: 0.12.
% <c> (optional) is the parameter that controls the strength of second-order contrast.
%   default: 0.99.
% <cache> (optional) is for speeding up execution.  can be [] or omitted.
%   if supplied (and not []), it must be the case that <stimulus> is identical to 
%   what it was when the <cache> was generated.
%
% compute the response of the SOC model to the images contained in <stimulus>.
% the output <response> will have dimensions F x F x N where F x F refers to 
% the grid of model outputs.  the output <cache> will contain various bits of information
% used to speed-up subsequent calls to socmodel.m.  some of the contents of <cache>
% may be useful for visualization purposes (see example below).
%
% the output returned by this function reflects not a single instance of the SOC model
% but in fact an array of instances of the SOC model that tile the image.  the 
% underlying idea is that we automatically position a distinct instance of the SOC model 
% at different locations on the contrast image.  this is the reason that the inputs to this 
% function do not include parameters specifying the center of the 2D Gaussian ---
% we actually use many 2D Gaussians.
%
% note that for simplicity, this function omits the gain parameter of 
% the SOC model (thus, the gain parameter is implicitly 1).
%
% history:
% - 2013/08/31 - initial version
%
% example:
% im = getsampleimage;
% im = imresize(im,[128 128]);
% [response,cache] = socmodel(im);
% figure; setfigurepos([100 100 600 300]);
% subplot(2,3,1); imagesc(im);
%   axis image tight; colormap(gray); colorbar; title('Original image');
% subplot(2,3,2); imagesc(cache.stimulus);
%   axis image tight; colormap(gray); colorbar; title('Resized, Range-Adjusted, Padded');
% subplot(2,3,3); imagesc(reshapesquare(cache.stimulus3));
%   axis image tight; colormap(gray); colorbar; title('Contrast image');
% subplot(2,3,4); imagesc(reshapesquare(cache.stimulus4));
%   axis image tight; colormap(gray); colorbar; title('Second-order contrast');
% subplot(2,3,5); imagesc(reshapesquare(cache.stimulus5));
%   axis image tight; colormap(gray); colorbar; title('Power-law nonlinearity');

% input
if ~exist('res','var') || isempty(res)
  res = [];  % deal with later
end
if ~exist('rng','var') || isempty(rng)
  rng = [min(stimulus(:)) max(stimulus(:)) mean(stimulus(:))];
end
if ~exist('efactor','var') || isempty(efactor)
  efactor = 1.2;
end
if ~exist('gparams','var') || isempty(gparams)
  gparams = {size(stimulus,1)/4 -1 1 8 2 .01 2 0};
end
if ~exist('r','var') || isempty(r)
  r = 1;
end
if ~exist('s','var') || isempty(s)
  s = 0.5;
end
if ~exist('sd','var') || isempty(sd)
  sd = 1.5;
end
if ~exist('spacing','var') || isempty(spacing)
  spacing = 1/1.5;
end
if ~exist('n','var') || isempty(n)
  n = 0.12;
end
if ~exist('c','var') || isempty(c)
  c = 0.99;
end
if ~exist('cache','var') || isempty(cache)
  cache = [];
end
if isempty(res)
  res = round(4*gparams{1});
end

%%%%%%%%%% RESIZE, RANGE, PAD

% if cache is empty, we have to preprocess the stimulus.
% otherwise, we have to preprocess if res or rng or efactor is different.
recompute1 = isempty(cache) || ~isequal(res,cache.res) || ~isequal(rng,cache.rng) || ~isequal(efactor,cache.efactor);
if recompute1

  % convert to single format to save memory
  cache.stimulus = single(stimulus);
  
  % resize if necessary
  if res >= size(cache.stimulus,1)
    fprintf('Since <res> is not smaller than the original stimulus resolution, we are omitting resizing.\n');
  else
    cache.stimulus = processmulti(@imresize,cache.stimulus,[res res],'cubic');
  end
  
  % deal with range
  cache.stimulus = normalizerange(cache.stimulus,0,1,rng(1),rng(2)) - normalizerange(rng(3),0,1,rng(1),rng(2));
  
  % calc final stimulus resolution after padding
  origres = size(cache.stimulus,1);
  padres = origres + round((efactor*origres - origres)/2)*2;
  
  % apply padding, filling in with zeros
  if padres ~= origres
    cache.stimulus = placematrix(zeros(padres,padres,size(cache.stimulus,3),'single'),cache.stimulus);
  end

  % record
  cache.res = res;
  cache.rng = rng;
  cache.efactor = efactor;
  cache.actualefactor = padres/origres;

end

%%%%%%%%%% APPLY GABOR FILTERS

% apply Gabor filters if we recomputed stimulus or if gparams is different
recompute2 = recompute1 || ~isequal(gparams,cache.gparams);
if recompute2

  % have to adjust the first argument of gparams
  gparamstemp = gparams;
  gparamstemp{1} = gparamstemp{1}*cache.actualefactor;

  % apply filters
  cache.stimulus2 = applymultiscalegaborfilters(squish(cache.stimulus,2)',gparamstemp{:});
  cache.stimulus2 = single(cache.stimulus2);  % ensure single format

  % record
  cache.gparams = gparams;

end

%%%%%%%%%% DIVISIVE NORMALIZATION AND SUM ACROSS ORIENTATION

% perform divisive normalization and sum across orientation
% if we recomputed stimulus2 or if r or s is different
recompute3 = recompute2 || ~isequal(r,cache.r) || ~isequal(s,cache.s);
if recompute3

  % calc
  numor = gparams{4};
  numph = gparams{5}; assert(numph==2);

  % do it
  cache.stimulus3 = sqrt(blob(cache.stimulus2.^2,2,2));  % complex-cell energy model
  popactivity = blob(cache.stimulus3,2,numor)/numor;
  cache.stimulus3 = blob(cache.stimulus3.^r,2,numor) ./ (s^r + popactivity.^r);

  % record
  cache.r = r;
  cache.s = s;
  
  % clean up
  clear popactivity;

end

%%%%%%%%%% PREPARE GAUSSIANS

% calc resolution of the contrast image
C = sqrt(size(cache.stimulus3,2));

% prepare gaussian filters if C or sd or spacing is different
recomputefilters = ~isfield(cache,'C') || ~isequal(C,cache.C) || ~isequal(sd,cache.sd) || ~isequal(spacing,cache.spacing);
if recomputefilters

  % generate the filters.  note that the base filter is empirically L1-normed after
  % thresholding at 0.01; this base filter is then positioned at different locations.
  % note that this is slightly different than the case where Gaussians
  % are generated at the ideal gain to achieve an L1-norm of 1.
  % the dimensionality of the filters is C*C x F*F where each column corresponds to
  % one Gaussian and the Gaussians occur on an F x F grid.
  [d,d,d,d,cache.filters] = applymultiscalegaussianfilters(randn(1,C^2),[sd],[spacing],.01,2);
  
  % make full and pre-compute which entries are non-zero
  cache.filters = full(cache.filters);
  cache.nz = cache.filters ~= 0;

  % record
  cache.C = C;
  cache.sd = sd;
  cache.spacing = spacing;

end

%%%%%%%%%% COMPUTE SECOND-ORDER CONTRAST

% compute second-order contrast if we recomputed stimulus3 or 
% if we recomputed the filters or if c is different
recompute4 = recompute3 || recomputefilters || ~isequal(c,cache.c);
if recompute4

  % compute average contrast within each Gaussian
  localavg = cache.stimulus3*cache.filters;  % N x F*F

  % loop over each Gaussian, computing second-order contrast
  cache.stimulus4 = zeros(size(cache.stimulus3,1),size(cache.filters,2),'single');
  fprintf('working');
  for p=1:size(cache.filters,2)
    statusdots(p,size(cache.filters,2));
    cache.stimulus4(:,p) = (cache.stimulus3(:,cache.nz(:,p)) - repmat(c * localavg(:,p),[1 sum(cache.nz(:,p))])).^2 * cache.filters(cache.nz(:,p),p);
  end
  fprintf('done.\n');
  
  % record
  cache.c = c;
  
  % clean up
  clear localavg;

end

%%%%%%%%%% APPLY POWER-LAW NONLINEARITY

% apply power-law nonlinearity if we recomputed stimulus4 or if n is different
recompute5 = recompute4 || ~isequal(n,cache.n);
if recompute5

  % do it
  cache.stimulus5 = cache.stimulus4 .^ n;

  % record
  cache.n = n;

end

%%%%%%%%%% PREPARE OUTPUT

% easy!
F = sqrt(size(cache.stimulus5,2));
response = reshape(cache.stimulus5',F,F,[]);
