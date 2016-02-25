function f = imreadmulti(files,converttogray)

% function f = imreadmulti(files,converttogray)
%
% <files> is a pattern that matches image files (see matchfiles.m)
% <converttogray> (optional) is whether to convert color images to grayscale.  default: 0.
%
% load and concatenate the images from the files named by <files>.
% we try to concatenate along the 3rd dimension if the first image is grayscale (2D)
% and the 4th dimension if the first image is color (3D).
%
% we assume all images have the same dimensions.
% we issue a warning if no files are named by <files>.

% input
if ~exist('converttogray','var') || isempty(converttogray)
  converttogray = 0;
end

% transform
files = matchfiles(files);

% check sanity
if length(files)==0
  warning('no file matches');
  f = [];
  return;
end

% do it
for p=1:length(files)

  % load image  
  loaded = imread(files{p});
  
  % convert?
  if converttogray && size(loaded,3) > 1
    loaded = rgb2gray(loaded);
  end

  % do it
  if p==1
    dim = ndims(loaded)+1;
    f = loaded;
  else
    f = cat(dim,f,loaded);
  end

end
