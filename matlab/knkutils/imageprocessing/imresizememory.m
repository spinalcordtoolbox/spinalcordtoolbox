function f = imresizememory(images,dres,dim)

% function f = imresizememory(images,dres,dim)
%
% <images> is A x B x images or A x B x 3 x images.
%   must be uint8 format.
% <dres> is the desired resolution [C D]
% <dim> is the dimension of <images> with different images
%
% perform some pre-processing and return an output in <f>.
% after completion of this call, the caller can delete
% <images> from memory.
%
%   AND THEN:
%
% function images = imresizememory(f)
%
% <f> is the output of the earlier call to imresizememory.m
%
% return the resized images in <images>.
%
% example:
% images = repmat(uint8(255*getsampleimage(1)),[1 1 10]);
% size(images)
% f = imresizememory(images,[100 100],3);
% clear images;
% images = imresizememory(f);
% size(images)

% the second case
if iscell(images)

  % calc
  file0 = images{1};
  sz = images{2};
  dres = images{3};
  dim = images{4};
  
  % handle pernicious case up front
  if isempty(sz)
    f = zeros([0 0],'uint8');
    delete(file0);
    return;
  end

  % calc
  newdims = sz;
  newdims(1:2) = dres;  % dimensions of the final matrix
  
  % do it
  f = zeros(newdims,'uint8');
  fprintf('imresizememory: processing images');
  for p=1:sz(dim)
    statusdots(p,sz(dim));
    im = uint8(imresize(double(loadbinary(file0,'uint8',sz,[p p])),dres));
    if dim==3
      f(:,:,p) = im;
    else
      f(:,:,:,p) = im;
    end
  end
  fprintf('done.\n');
  delete(file0);

% the first case
else

  file0 = tempname;
  fprintf('imresizememory: saving images to disk...');
  savebinary(file0,'uint8',images);
  fprintf('done.\n');
  
  % handle crazy case
  if isempty(images)
    sz = [];
  else
    sz = size(images);
  end
  
  % finish up
  f = {file0 sz dres dim};

end
