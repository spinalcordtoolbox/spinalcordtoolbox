function [f,filenames] = concatimages(files,order)

% function [f,filenames] = concatimages(files,order)
%
% <files> is a pattern that matches image files (see matchfiles.m)
% <order> is a cell vector of things, each of which specifies indices
%   of images to put on the same row
%
% load and concatenate the images from <files>.
% we do this by first putting images on rows and
%   then concatenating the rows together.
%   our anchor point is always the upper-left corner.
% we return a uint8 image in <f>.
% we also return a cell vector of the matched filenames in <filenames>.
% we assume that images are either all grayscale or all color.
% we issue a warning if no files are named by <files>.
%
%   OR
%
% function [f,filenames] = concatimages(files)
%
% <files> is a cell vector {A B C ...} where A, B, C, ... are each
%   cell vectors of patterns that match image files (see matchfiles.m)
%
% this is like the previous scheme, except that we have a different
% format for <files>.  each A, B, C, ... corresponds to a row
% in the output image.

if exist('order','var')

  % transform
  filenames = matchfiles(files);
  
  % check sanity
  if length(filenames)==0
    warning('no file matches');
    f = [];
    return;
  end
  
  % do it
  f = [];
  for pp=1:length(order)
    im0 = [];
    for qq=1:length(order{pp})
      im0 = placematrix2(im0,imread(filenames{order{pp}(qq)}),[1 size(im0,2)+1 1]);
    end
    f = placematrix2(f,im0,[size(f,1)+1 1 1]);
  end
  f = uint8(f);

else

  % do it
  f = []; filenames = {};
  for pp=1:length(files)
    im0 = [];
    for qq=1:length(files{pp})
      temp = matchfiles(files{pp}{qq});
      for rr=1:length(temp)
        filenames = [filenames temp{rr}];
        im0 = placematrix2(im0,imread(temp{rr}),[1 size(im0,2)+1 1]);
      end
    end
    f = placematrix2(f,im0,[size(f,1)+1 1 1]);
  end
  f = uint8(f);

  % check sanity
  if length(filenames)==0
    warning('no file matches');
    f = [];
    return;
  end

end
