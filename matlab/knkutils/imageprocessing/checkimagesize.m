function [dim,files] = checkimagesize(patterns)

% function [dim,files] = checkimagesize(patterns)
%
% <patterns> is where to look for image files (see matchfiles.m)
%
% return <dim> as N x 2 with the pixel dimensions of the N images.
% the first column gives the number of rows; the second column gives the number of columns.
% the dimensions are determined using the "identify" utility from ImageMagick.
% also, return <files> with paths to the matched files.
% 
% as we process, we report to stdout.
%
% note that in ImageMagick, A x B means A is the width and B is the height.
% note that we do NOT use this convention!

% get location of image files
files = matchfiles(patterns);
numfiles = length(files);

% do it
dim = zeros(numfiles,2);
for p=1:numfiles
  [status,result] = unix(['identify ' files{p}]);
  assert(status==0);
  temp = regexp(result,'(?<a>\d+)x(?<b>\d+)','names');
  dim(p,:) = [str2double(temp(1).b) str2double(temp(1).a)];  % NOTE THE FLIP

  % report
  fprintf(1,'%d x %d, file: %s (%d of %d)\n',dim(p,1),dim(p,2),files{p},p,numfiles);
end
