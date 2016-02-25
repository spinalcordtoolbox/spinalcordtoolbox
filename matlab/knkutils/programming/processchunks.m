function f = processchunks(fun,m,dim,num)

% function f = processchunks(fun,m,dim,num)
%
% <fun> is a function that accepts M x N and outputs 1 x N
% <m> is a matrix of any dimensions
% <dim> is the dimension of <m> that is of interest
% <num> (optional) is the number of chunks to divide the
%   problem into.  default: 100.
%
% sometimes, if memory is limited, it is desirable to 
% break up a problem into several pieces so as not to
% overrun the available memory.  this function allows
% you to do that easily.
%
% example:
% xx = randn(100,100,500);
% test = std(xx,[],3);
% test2 = processchunks(@(x) std(x,[],1),xx,3);
% isequal(test,test2)

% input
if ~exist('num','var') || isempty(num)
  num = 100;
end

% calc
msize = size(m);
msize(dim) = 1;  % the final output should have only one element along <dim>
m = reshape2D(m,dim);  % <m> is now M x N

% do it
chunks = chunking(1:size(m,2),ceil(size(m,2)/num));
f = zeros(1,size(m,2));
for p=1:length(chunks)
  f(chunks{p}) = feval(fun,m(:,chunks{p}));
end

% calc
f = reshape2D_undo(f,dim,msize);
