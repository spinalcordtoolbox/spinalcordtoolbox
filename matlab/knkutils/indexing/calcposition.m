function f = calcposition(list,x)

% function f = calcposition(list,x)
%
% <list> is a vector with unique elements (must be positive integers)
% <x> is a vector whose elements are in <list>.
%   elements can be in any order and repeats are okay.
%
% return a vector the same length as <x> with indices relative to <list>.
%
% example:
% isequal(calcposition([5 3 2 4],[2 2 5]),[3 3 1])

% construct a vector that gives the correct index for X if you extract the Xth element
xrank = NaN*zeros(1,max(list));  % if max(list) is big, this is ouch
xrank(list) = 1:length(list);

% get the answers
f = xrank(x);

% sanity check
assert(~any(isnan(f)),'<list> does not subsume <x>');




% NICER, BUT SLOWER:
% % init
% f = zeros(size(x));
% % do it
% xu = union(x,[]);
% for p=1:length(xu)
%   temp = find(list==xu(p));
%   assert(~isempty(temp),'<list> does not subsume <x>');
%   f(x==xu(p)) = temp;  % POTENTIALLY SLOW.  see commented code below for a faster but less general solution
% end
