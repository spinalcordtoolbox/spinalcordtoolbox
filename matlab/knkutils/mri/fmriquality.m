function fmriquality(vols,volsize,figuredir,seed,numrandom,knobs,inplaneextra)

% function fmriquality(vols,volsize,figuredir,seed,numrandom,knobs,inplaneextra)
%
% <vols> is a 4D volume (X x Y x Z x T) or a cell vector of 4D volumes.
%   these volumes should be suitable for interpretation as double
%   (though they do not need to be passed in as double).  the 
%   first three dimensions must be consistent across cases.
% <volsize> is a 3-element vector with the voxel size in mm.
%   the first two voxel dimensions must be the same size.
% <figuredir> is the directory to write figures to.
% <seed> (optional) is the rand state to use.  default: 0.
% <numrandom> (optional) is the number of volumes to randomly select.  default: 100.
% <knobs> (optional) is [A B C D E] with various constants used in
%   the analysis.  see below for details.  default: [99 1/4 1/10 5 10].
% <inplaneextra> (optional) is {A B} where A is [R C] with the matrix size
%   of the in-plane volume and B is [Rsz Csz] with the corresponding voxel
%   sizes in mm.  if this input is supplied, we write out an additional figure
%   (see below).
%
% write out a number of figures that illustrate the spatial quality of the
% fMRI volumes in <vols>.  if you have matlabpool on, we take advantage of parfor.
% note that there is some overlap of this function with homogenizevolumes.m.
%
% here are the details:
% - meanvol.png is the mean across all volumes.  the range is contrast-normalized.
% - meanvolMATCH.png is the mean across all volumes, matched to the in-plane resolution and
%   field-of-view using cubic interpolation.  this is written only if <inplaneextra> 
%   is supplied.  the range is contrast-normalized.
% - HIGH refers to the 99th (i.e. Ath) percentile of the mean volume.
% - brainmask.png is voxels we deem to be "brain".  this is determined by selecting
%   voxels in the mean volume that are at least 1/4 (i.e. B) of HIGH.
% - stdvol.png is the standard deviation across all volumes.  the range is [0 1/10*HIGH]
%   (i.e. [0 C*HIGH]).
% - stdvolhist.png has a histogram of the standard deviation volume.
% - randomvolumes/image%04d.png are <numrandom> randomly selected volumes.  the same range
%   is used for all volumes.  (if there are less than <numrandom> volumes, we just take
%   as many volumes as there are; the order will nevertheless be random.)
% - contours_all.png has the half-height contour for all slices.  we superimpose
%   the results for the <numrandom> randomly selected volumes on the same figure.
% - contours_slice%02d.png has the half-height contour for individual slices.  we
%   superimpose results for the <numrandom> randomly selected volumes on the same figure. 
% 
% details on contour determination:
% - before determining contours, we homogenize each volume by fitting a 3D polynomial
%   of degree up to 5 (i.e. D) to the "brain" voxels and then dividing the whole volume by 
%   the fitted polynomial.  to ensure reasonable results, before the division the
%   fitted polynomial is massaged such that any negative values are set to Inf (so that
%   they result in 0 after division) and values are truncated at 1/10 (i.e. 1/E) of the 
%   maximum value in the fitted polynomial (so that the maximum scaling to be applied is 
%   10 (i.e. E)).
% - after polynomial division, we proceed to the contour calculation.  first,
%   NaN values are set to 0.  then, we calculate the contour at a level of 0.5.
%   then, we render the image.  finally, we repeat the process for each volume, and 
%   then average the rendered images.  we display the average rendered image using a 
%   hot colormap.  pixels that are pure white indicate that the contours from all
%   volumes coincided at those pixels.  when misregistration exists, it will show
%   up as something like blurry, fat borders that are dark red.
%
% history:
% 2011/07/28 - calculate stdvol using processchunks to save memory
% 2011/07/28 - to save memory, calculate stdvol using only single precision
% 2011/04/17 - implement <inplaneextra>
% 2011/03/26 - change default to [99 1/4 1/10 5 10]
%
% example:
% vols = getsamplebrain(4);
% fmriquality(vols,[2.5 2.5 2.5],'test');

% MAYBE TODO:
% * draw a line, show me the spread in intersections.
% *Êtake the contour images and calculate MI?  somehow quantify
% * and maybe save contours for later.?
% * what about MI metric compared to inplane
% what about temporal SNR?

% input
if ~exist('seed','var') || isempty(seed)
  seed = 0;
end
if ~exist('numrandom','var') || isempty(numrandom)
  numrandom = 100;
end
if ~exist('knobs','var') || isempty(knobs)
  knobs = [99 1/4 1/10 5 10];
end
if ~exist('inplaneextra','var') || isempty(inplaneextra)
  inplaneextra = [];
end
if ~iscell(vols)
  vols = {vols};
end

% make figure dir
mkdirquiet(figuredir);

% calc
fprintf('performing simple calculations...');
numinrun = cellfun(@(x) size(x,4),vols);  % number of volumes in each run
cumtotalnum = cumsum(numinrun);           % total volumes so far
totalnum = sum(numinrun);                 % total number of volumes
xyzsize = sizefull(vols{1},3);            % [X Y Z]
numvoxels = prod(xyzsize);                % total number of voxels
meanvol = mean(catcell(4,vols),4);        % mean volume.  could have NaNs.
stdvol = double(processchunks(@(x) std(single(x),[],1),catcell(4,vols),4));  % std volume.  could have NaNs.
highval = prctile(meanvol(:),knobs(1));   % what is a high signal intensity value?
lowval = highval*knobs(2);                % what is a low value?
brainmask = meanvol >= lowval;            % where are brain voxels?
stdmx = highval*knobs(3);                 % what is a good upper limit for std?
todo = picksubset(1:totalnum,numrandom,seed);  % figure out which ones to process
numtodo = length(todo);                   % total number to process
fprintf('done.\n');

% write out brain mask and other simple things
fprintf('writing out simple volumes...');
imwrite(uint8(255*makeimagestack(brainmask,[0 1])),gray(256),sprintf('%s/brainmask.png',figuredir));
imwrite(uint8(255*makeimagestack(meanvol,1)),gray(256),sprintf('%s/meanvol.png',figuredir));
if ~isempty(inplaneextra)
  imwrite(uint8(255*makeimagestack( ...
    processmulti(@imresizedifferentfov,meanvol,volsize(1:2),inplaneextra{1},inplaneextra{2}), ...
    1)),gray(256),sprintf('%s/meanvolMATCH.png',figuredir));
end
imwrite(uint8(255*makeimagestack(stdvol,[0 stdmx])),jet(256),sprintf('%s/stdvol.png',figuredir));
%%imwrite(uint8(255*makeimagestack(20 - (stdvol./meanvol * 100),[0 20])),hot(256),sprintf('%s/stdvolALT.png',figuredir));
fprintf('done.\n');

% write out random volumes
fprintf('writing out random volumes...');
collect = zeros([xyzsize numtodo]);
for p=1:numtodo
  wh = firstel(find(todo(p) <= cumtotalnum));
  ix = todo(p) - sum(numinrun(1:wh-1));
  collect(:,:,:,p) = vols{wh}(:,:,:,ix);  % implicit conversion to double
end
viewmovie(collect,sprintf('%s/randomvolumes/image%%04d',figuredir));
clear collect;
fprintf('done.\n');

% write out figure illustrating stdvol stuff
fprintf('writing out stdvol histogram...');
figureprep; hold on;
hist(stdvol(:),linspace(0,stdmx,50));
xlabel('Standard deviation (raw units)'); ylabel('Frequency');
title(sprintf('median = %.1f, median (brain mask) = %.1f',nanmedian(stdvol(:)),nanmedian(stdvol(brainmask))));
figurewrite('stdvolhist',[],[],figuredir);
fprintf('done.\n');

% prepare volumes by dividing by a fitted polynomial
fprintf('homogenizing volumes...');
vols = cellfun(@(x) int16(10000 * x),homogenizevolumes(vols,[knobs(1) knobs(2) knobs(4) knobs(5)]),'UniformOutput',0);
    % if vols is int16, then this should ensure a good range
    % this step is a bit weird, since we impose int16 even if vols wasn't int16.  but should be okay.
fprintf('done.\n');

% write out contour figures
  % first do all slices in one figure
fprintf('writing out contours (all)');
sl = [];
fun = @(wh,ix,sl) makeimagestack(double(vols{wh}(:,:,:,ix)),[],j);
filename = '%s/contours_all%d.png';
fmriquality_helper;
fprintf('done.\n');
  % then, do each slice in separate figure
for sl=1:xyzsize(3)
  fprintf('writing out contours (slice %d)',sl);
  fun = @(wh,ix,sl) double(vols{wh}(:,:,sl,ix));
  filename = '%s/contours_slice%02d.png';
  fmriquality_helper;
  fprintf('done.\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% JUNK:

% % 2. calculate coefficient of variation.  write volume and a figure window.
% coeffvariation = negreplace(zerodiv(stdvol,meanvol,NaN,0) * 100,NaN);
% imwrite(uint8(255*makeimagestack(coeffvariation,[0 5])),jet(256),sprintf('%s/coeffvariation.png',figuredir));

% check = interp1(temp(:,1),1:64,0.5,'linear')
