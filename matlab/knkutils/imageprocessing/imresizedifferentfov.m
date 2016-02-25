function f = imresizedifferentfov(im,imsize,res,ressize)

% function f = imresizedifferentfov(im,imsize,res,ressize)
%
% <im> is a 2D image of dimensions R x C.  can have multiple images
%   along the third dimension, in which case we process each image
%   independently.
% <imsize> is [Rsz Csz] with the size of a pixel along each dimension
% <res> is the desired image resolution R2 x C2
% <ressize> is [R2sz C2sz] with the size of a pixel along each dimension
%
% the key feature of this routine is that it allows the original and
% target images to have different fields-of-view.  we assume that the
% original and target images share the same center.  our strategy is to 
% first interpolate the original image using pixels at least as small as 
% the smallest pixel size in <imsize>.  this interpolation is done to
% obtain exactly the desired field-of-view as specified by the combination
% of <res> and <ressize>, and is performed using cubic interpolation.
% then, we simply take the result and imresize it to <res> using cubic
% interpolation.
%
% example:
% im = getsamplebrain(2);
% im = im(:,:,10);
% im2 = imresizedifferentfov(im,[.75 .75],[64 32],[2 4]);
% figure; imagesc(im);
% figure; imagesc(im2);

% calc
gran = min(imsize);  % when we do the intermediate interpolation, need pixels at least this small
resfov = res .* ressize;  % this is the FOV for the final image
center0 = [(1+size(im,1))/2 (1+size(im,2))/2];  % the center of the image in the original matrix units

% calc more
rnum = ceil(resfov(1)/gran);  % need this many pixels in the rows of the intermediate interpolation
cnum = ceil(resfov(2)/gran);  % need this many pixels in the columns of the intermediate interpolation
irsize = resfov(1)/rnum;      % size of pixel in the row direction in the intermediate interpolation
icsize = resfov(2)/cnum;      % size of pixel in the column direction in the intermediate interpolation

% calc even more (the x- and y-positions of the intermediate interpolation)
xi = linspace(center0(2) - (cnum/2 * icsize - icsize/2)/imsize(2), ...
              center0(2) + (cnum/2 * icsize - icsize/2)/imsize(2),cnum);
yi = linspace(center0(1) - (rnum/2 * irsize - irsize/2)/imsize(1), ...
              center0(1) + (rnum/2 * irsize - irsize/2)/imsize(1),rnum);

% create intermediate interpolation and then imresize it
f = zeros(res(1),res(2),size(im,3));
for p=1:size(im,3)
  f(:,:,p) = imresize(interp2(im(:,:,p),xi,yi(:),'*cubic'),res,'cubic');
end
