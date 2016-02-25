function f = processmulti(fun,varargin)

% function f = processmulti(fun,varargin)
%
% <fun> is a function that expects a 2D image as the first argument
% <varargin> is a set of inputs.  the first input should be one or multiple images
%   concatenated along the third dimension.  the remaining inputs are the
%   same as expected by <fun>.
%
% apply <fun> to each image and concatenate the results together.
%
% example:
% isequal(size(processmulti(@blkproc,randn(10,10,3),[5 5],@(x) mean(x(:)))),[2 2 3])
% isequal(size(processmulti(@colfilt,randn(10,10,3),[2 2],'sliding',@(x) mean(x,1))),[10 10 3])
% isequal(size(processmulti(@imresize,randn(10,10,3),[20 20])),[20 20 3])

  prev = warning('query');  % make sure stupid warning about obsolete blkproc doesn't come on
  warning('off');
f = [];
for p=1:size(varargin{1},3)
  if p==1
    f = feval(fun,varargin{1}(:,:,p),varargin{2:end});
    f = placematrix2(zeros([size(f) size(varargin{1},3)],class(f)),f);
  else
    f(:,:,p) = feval(fun,varargin{1}(:,:,p),varargin{2:end});
  end
end
  warning(prev);
