function f = makeimagestack_wrapper(varargin)

% function f = makeimagestack_wrapper(varargin)
%
% input arguments are the same as for makeimagestack.m, except that
% <m> can be a true 4D matrix.
%
% we pass 3D slices of <m> to makeimagestack.m and concatenate the results
% along the third dimension (thus, we assume that the output of makeimagestack.m
% consists of one set of images embedded in a 2D matrix).  this function is
% useful for processing color images.
%
% example:
% figure; imagesc(makeimagestack_wrapper(rand(10,10,5,3),0,1,[1 5]));

f = [];
for p=1:size(varargin{1},4)
  f = cat(3,f,makeimagestack(varargin{1}(:,:,:,p),varargin{2:end}));
end
