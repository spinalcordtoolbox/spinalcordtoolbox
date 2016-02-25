function vol = preconditionvolume(vol,thresh,polydeg,polythresh,wantfigs)

% function vol = preconditionvolume(vol,thresh,polydeg,polythresh,wantfigs)
%
% <vol> is a 3D volume (there must be at least 2 elements along each dimension)
% <thresh> (optional) is [A B] where A is a number of percentiles and B is a
%   multiplier.  for example, [40 4] means find the median, add 4 times the 
%   difference between the median and the 90th percentile to get the upper
%   bound, and subtract 4 times the difference between the median and the
%   10th percentile to get the lower bound.  this input determines how we
%   clip outliers from <vol>.  default: [40 4].  can be [], in which case
%   we omit the clipping of outliers.
% <polydeg> (optional) is the maximum polynomial degree desired for fitting
%   3D polynomial basis functions.  default: 2.
% <polythresh> (optional) is [C D] where C is a percentile and D is a fraction.
%   for example, [99 1/10] means find the 99th percentile and then multiply
%   it by 1/10 in order to get the threshold above which we consider voxels
%   for fitting the polynomials.  default: [99 1/10].
% <wantfigs> (optional) is whether to make the figure windows.  default: 1.
%
% pre-condition <vol> by clipping outlier voxels and by removing low-frequency
% signal variations.  more specifically, we perform the following steps:
%   (1) clip outliers from <vol> based on <thresh> (see clipoutliers.m for details)
%   (2) fit 3D polynomial basis functions up to degree <polydeg>, considering only
%       voxels that pass <polythresh>.  (actually, to increase execution speed,
%       we consider only a small random subset of voxels that pass <polythresh>;
%       see below for details).
%   (3) evaluate the polynomial model (i.e. weighted combination of 3D polynomial
%       basis functions) over all voxels (not just the ones fitted).  then divide
%       <vol> by the polynomial model.
%   (4) since division can result in values that blow up, clip outliers one more
%       time based on <thresh>.
% we return the final version of <vol>.
%
% we generate three figure windows.  in the first figure window, we show the
% volume before and after the first clipping step.  red lines indicate the
% clipping cutoff points.  the percentage of voxels that are clipped are reported
% in the title.  in the second figure window, we show four images.  the top two
% images are the volume before polynomial division and the volume after polynomial
% division.  the bottom two images are the polynomial model and the mask indicating
% which voxels were considered (before subsetting) when fitting the polynomial model.
% note that in general, each of the images has been contrast-normalized, so don't 
% attempt to make cross-image comparisons in that way.  only up to 16 slices from 
% the third dimension of the volume are shown; these slices are equally spaced 
% (as best as possible) between the first and last slice along the third dimension.
% in the third figure window, we show horizontal and vertical profiles through the
% volume before polynomial division and through the volume after polynomial division.
% the profiles cut through the middle of each slice, and we show only the slices 
% that are also shown in the second figure window.  the median across the shown 
% profiles is indicated by a thick black line.
% 
% for faster execution, in various steps of the process we operate on random 
% subsets of maximum size 10000 (these subsets are random but deterministic).
% these steps include calculating percentiles and fitting the 3D polynomial
% basis functions.
%
% example:
% vol = getsamplebrain(2);
% volB = preconditionvolume(vol);
% figure; imagesc(makeimagestack(vol,5));
% figure; imagesc(makeimagestack(volB,5));

% internal constants
subnum = 10000;  % this number is plenty enough to estimate percentiles and to fit the poly model
maxslices = 16;  % show at most this number of slices

% which slices should we show?
if size(vol,3) > maxslices
  sliceix = round(linspace(1,size(vol,3),maxslices));
else
  sliceix = 1:size(vol,3);
end

% what is the total number of voxels?
tot = numel(vol);

% ensure double
vol = double(vol);

% input
if ~exist('thresh','var') || isempty(thresh)
  thresh = [40 4];
end
if ~exist('polydeg','var') || isempty(polydeg)
  polydeg = 2;
end
if ~exist('polythresh','var') || isempty(polythresh)
  polythresh = [99 1/10];
end
if ~exist('wantfigs','var') || isempty(wantfigs)
  wantfigs = 1;
end

% clip outliers
if ~isempty(thresh)
  [vol,pmx,pmn,highcut,lowcut] = clipoutliers(vol,subnum,thresh);
end

% first figure
if wantfigs && ~isempty(thresh)
  [nn,bins] = hist(vol(:),100);
  figure; setfigurepos([100 100 600 200]);
  subplot(1,2,1); hold on;
  bar(bins,nn,1);
  straightline([pmn pmx],'v','r-');
  title('histogram (before)');
  subplot(1,2,2); hold on;
  hist(vol(:),100);
  straightline([pmn pmx],'v','r-');
  title(sprintf('histogram (after), lowcut %.1f%%, highcut %.1f%%',lowcut/tot*100,highcut/tot*100));
  drawnow;
end

% calc
horiz1 = squish(vol(round(end/2),:,sliceix),2);
vert1 = squish(vol(:,round(end/2),sliceix),2);

% fit polynomial model
mask = vol > prctile(picksubset(vol,subnum),polythresh(1))*polythresh(2);
maskB = copymatrix(zeros(size(vol)),picksubset(find(mask),subnum),1);
params = fit3dpolynomialmodel(vol,maskB,polydeg);
polymodel = reshape(constructpolynomialmatrix3d(size(vol),1:numel(vol),polydeg,params),size(vol));

% calc figure stuff
if wantfigs
  rng = prctile(picksubset(vol,subnum),[1 99]);
  im1 = makeimagestack(vol(:,:,sliceix),rng);  % SLICES FROM ORIGINAL
  im2 = makeimagestack(mask(:,:,sliceix),[0 1]);  % SLICES FROM MASK
  im3 = makeimagestack(polymodel(:,:,sliceix),rng);  % SLICES FROM POLY FIT
end

% divide out and threshold again
vol = zerodiv(vol,polymodel,[],0);
if ~isempty(thresh)
  vol = clipoutliers(vol,subnum,thresh);
end

% make new figures
if wantfigs
  rngB = prctile(picksubset(vol,subnum),[1 99]);
  im4 = makeimagestack(vol(:,:,sliceix),rngB);  % SLICES FROM FINAL
  horiz2 = squish(vol(round(end/2),:,sliceix),2);
  vert2 = squish(vol(:,round(end/2),sliceix),2);

  figure; setfigurepos([.1 .1 .8 .8]); hold on;
  setaxispos([0 0 1 1]);
  imagesc([im1 im4; im3 im2],[0 1]);
  axis off image ij;
  colormap(gray(256));
  title('before and after; poly and mask');
  drawnow;

  figure; setfigurepos([200 100 600 300]);
  subplot(2,2,1); hold on;
  plot(horiz1);
  plot(median(horiz1,2),'k-','LineWidth',2);
  title('horizontal profiles (before)');
  subplot(2,2,2); hold on;
  plot(horiz2);
  plot(median(horiz2,2),'k-','LineWidth',2);
  title('horizontal profiles (after)');
  subplot(2,2,3); hold on;
  plot(vert1);
  plot(median(vert1,2),'k-','LineWidth',2);
  title('vertical profiles (before)');
  subplot(2,2,4); hold on;
  plot(vert2);
  plot(median(vert2,2),'k-','LineWidth',2);
  title('vertical profiles (after)');
  drawnow;
end
