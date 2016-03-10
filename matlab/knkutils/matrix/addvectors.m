function f = addvectors(varargin)

% function f = addvectors(x1,x2,x3,...)
%
% <x1>,<x2>,<x3>... are horizontal vectors
%
% add together, assuming zeros to the right as necessary.
%
% example:
% a = [1 2 3];
% b = [3 4];
% isequal(addvectors(a,b),[4 6 3])

% get length of each argument
lengths = zeros(1,nargin);  % could use cellfun but that's slow and we need speed!
for p=1:nargin
  lengths(p) = length(varargin{p});
end

% init
f = zeros(nargin,max(lengths));

% populate
for p=1:nargin
  f(p,1:lengths(p)) = varargin{p};
end

% sum
f = sum(f,1);





% OLD IMPLEMENTATION
% % written like this for speed
% lx = length(x);
% ly = length(y);
% if lx > ly
%   f = x;
%   if ly~=0
%     f(1:ly) = f(1:ly) + y;
%   end
% else
%   f = y;
%   if lx~=0
%     f(1:lx) = f(1:lx) + x;
%   end
% end

% OLD IMPLEMENTATION
% f = zeros(1,max(length(x),length(y)));
% if ~isempty(x)
%   f(1:length(x)) = x;
% end
% if ~isempty(y)
%   f(1:length(y)) = f(1:length(y)) + y;
% end
