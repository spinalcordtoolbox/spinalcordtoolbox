function f = processmulti1D(fun,varargin)

% function f = processmulti1D(fun,varargin)
%
% <fun> is a function that expects a row vector as the first argument
% <varargin> is a set of inputs.  the first input should be one or multiple row vectors
%   concatenated along the first dimension.  the remaining inputs are the
%   same as expected by <fun>.
%
% apply <fun> to each vector and concatenate the results together (along the first dimension).
%
% example:
% isequal(processmulti1D(@mean,[1 2 3; 4 5 6]),[2; 5])

  prev = warning('query');  % make sure stupid warning about obsolete blkproc doesn't come on
  warning('off');
f = [];
for p=1:size(varargin{1},1)
  if p==1
    f = feval(fun,varargin{1}(p,:),varargin{2:end});  % this ensures class is respected
  else
    f = cat(1,f,feval(fun,varargin{1}(p,:),varargin{2:end}));
  end
end
  warning(prev);
