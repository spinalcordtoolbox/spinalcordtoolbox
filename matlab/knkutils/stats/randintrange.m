function f = randintrange(x,y,sz,wantnorepeats)

% function f = randintrange(x,y,sz,wantnorepeats)
% 
% <x> and <y> are integers such that <x> <= <y>
% <sz> (optional) is a matrix size.  default: [1 1].
% <wantnorepeats> (optional) is whether to enforce that
%   no successive repeats are to be returned.  be careful that 
%   the implementation of this feature may be slow.  default: 0.
%
% return random integers in the range [<x>,<y>] inclusive.
%
% example:
% randintrange(-2,5)

% input
if ~exist('sz','var') || isempty(sz)
  sz = [1 1];
end
if ~exist('wantnorepeats','var') || isempty(wantnorepeats)
  wantnorepeats = 0;
end

% do it
if wantnorepeats
  % SLOW AND UGLY!!!
  f = zeros(sz);
  prev = NaN;
  for p=1:numel(f)
    while 1
      f(p) = x + floor(rand*(y-x+1));
      if f(p) ~= prev
        prev = f(p);
        break;
      end
    end
  end
else
  f = x + floor(rand(sz)*(y-x+1));
end
