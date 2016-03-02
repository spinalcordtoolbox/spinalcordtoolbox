function varargout = chunking(varargin)

% function f = chunking(v,num)
%
% <v> is a vector
% <num> is desired length of a chunk
%
% return a cell vector of chunks.  the last vector 
% may have fewer than <num> elements.
%
% example:
% isequal(chunking(1:5,3),{1:3 4:5})
%
% OR
% 
% function [f,xbegin,xend] = chunking(v,num,n)
%
% <v> is a vector
% <num> is length of a chunk
% <n> is chunk number desired
%
% return the desired chunk in <f>.
% also return the beginning and ending indices associated with 
% this chunk in <xbegin> and <xend>.
%
% example:
% isequal(chunking([4 2 3],2,2),3)

switch length(varargin)

case 2
  v = varargin{1};
  num = varargin{2};
  
  f = {};
  for p=1:ceil(length(v)/num)
    f{p} = v((p-1)*num+1 : min(length(v),p*num));
  end
  
  varargout = {f};

case 3
  v = varargin{1};
  num = varargin{2};
  n = varargin{3};
  
  xbegin = (n-1)*num+1;
  xend = min(length(v),n*num);
  f = v(xbegin:xend);
  
  varargout = {f xbegin xend};

end
