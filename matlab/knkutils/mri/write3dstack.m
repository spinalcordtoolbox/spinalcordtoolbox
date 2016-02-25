function write3dstack(prefix,vol,volsize,inplaneres,sliceskip,wantnorm,rots,cmap)

% function write3dstack(prefix,vol,volsize,inplaneres,sliceskip,wantnorm,rots,cmap)
%
% <prefix> is a file location.  if [], we open up three figure windows instead.
% <vol> is a 3D matrix
% <volsize> is a 3-element vector with the size in mm 
%   of each of the three dimensions
% <inplaneres> (optional) is desired size in mm for the pixels
%   in the written images.  default: 0.5.
% <sliceskip> (optional) is how many mm to skip in the slice
%   dimension.  can also be [A B] where A is the mm to skip
%   and B is the desired fatness of each slice in mm.  if B is not
%   supplied, we default to 0, which in effect means to just
%   interpolate as usual.  if B is supplied, we attempt to 
%   go up and down by B/2 and whatever slice centers we
%   encounter, we average those slices together for the final
%   output.  (if we go outside the range of the original volume,
%   we just act as if we replicate the edge slices.)  default: 15.
% <wantnorm> (optional) is the <wantnorm> input to makeimagestack.m.
%   default: 1.
% <rots> (optional) is a 3-element vector indicating number of CCW
%   rotations to apply to each of the three views.  default: [0 1 1].
% <cmap> (optional) is the colormap to use.  default: gray(256).
%
% the point of this function is to visualize the potentially large
% 3D matrix <vol>.  we do this by extracting slices in three
% orientations from <vol>.  to avoid images that are too big,
% we extract slices only every <sliceskip> mm (rounded to the 
% nearest slice and then potentially averaged across several slices
% if the B input of <sliceskip> is provided), starting from the 
% first slice.  also, given the extracted slices, we automatically 
% ensure that the returned pixels are square in size (for easy 
% viewing) by interpolating through the extracted slices using 
% cubic interpolation (actually, the pixels may be slightly 
% off-square since we have to round to the nearest 
% whole number of pixels for each dimension).
% 
% three sets of slices are returned: the first set reflects dimensions
% 1 and 2, treating dimension 3 as the slice dimension; the second set 
% reflects dimensions 1 and 3, treating dimension 2 as the slice dimension;
% and the third set reflects dimensions 2 and 3, treating dimension 1 as
% the slice dimension.  note that the returned slices have a deterministic 
% in-plane rotation applied to them, and each set of slices is 
% separately contrast-normalized.
%
% the three sets of slices are written to files <prefix>_1.png, 
% <prefix>_2.png, and <prefix>_3.png.
%
% example:
% vol = getsamplebrain(3);
% write3dstack('test',vol,[1 1 1]);

% input
if ~exist('inplaneres','var') || isempty(inplaneres)
  inplaneres = 0.5;
end
if ~exist('sliceskip','var') || isempty(sliceskip)
  sliceskip = 15;
end
if ~exist('wantnorm','var') || isempty(wantnorm)
  wantnorm = 1;
end
if ~exist('rots','var') || isempty(rots)
  rots = [0 1 1];
end
if ~exist('cmap','var') || isempty(cmap)
  cmap = gray(256);
end
if length(sliceskip)==1
  sliceskip = [sliceskip 0];
end

% calc
voldim = sizefull(vol,3);       % matrix size
volfov = voldim .* volsize;     % FOV along each dimension

% do it
todo = {[1 2] [1 3] [2 3]};
for qq=1:length(todo)
  in = todo{qq};
  sl = setdiff(1:3,in);

  % extract slices and resample
  dsize = round(volfov(in)/inplaneres);   % desired image resolution for the two image dimensions, [A B]
  skip = round(sliceskip(1)/volsize(sl));    % number of slices to skip, N
  numup = round(sliceskip(2)/2 /volsize(sl));
  final = 0;
  for pp=-numup:numup
    ix = {':' ':' ':'}; ix{sl} = (1:skip:voldim(sl)) + pp;
    ix{sl} = min(ix{sl},voldim(sl));
    ix{sl} = max(ix{sl},1);
    final = final + processmulti(@imresizedifferentfov,permute(subscript(vol,ix),[in sl]),volsize(in),dsize,volfov(in)./dsize);
  end
  final = final / (2*numup+1);

  % write out
  temp = makeimagestack(rotatematrix(final,1,2,rots(qq)),wantnorm);
  if isempty(prefix)
    drawnow; figure; imagesc(temp,[0 1]); colormap(cmap);
  else
    imwrite(uint8(255*temp),cmap,sprintf('%s_%d.png',prefix,qq));
  end

end
