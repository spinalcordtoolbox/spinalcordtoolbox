function [f,idx] = filterout(m,elt,mode)

% function [f,idx] = filterout(m,elt,mode)
%
% <m> is a matrix
% <elt> is a number (can be NaN when <mode>==0)
% <mode> (optional) is
%   0 means ~=
%   1 means >
%   2 means <
%   3 means >=
%   4 means <=
%   default: 0.
%
% return <m> as a horizontal vector.  when <mode> is
%   0, return elements ~= <elt>
%   1, return elements > <elt>
%   2, return elements < <elt>
%   3, return elements >= <elt>
%   4, return elements <= <elt>
% also return <idx> as a logical matrix the same size as <m> 
% indicating which ones we pulled out.
%
% example:
% isequal(filterout([1 2 3],2),[1 3])

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% do it
switch mode
case 0
  if isnan(elt)
    idx = ~isnan(m);
  else
    idx = m~=elt;  % note that Inf and -Inf are okay here
  end
case 1
  idx = m>elt;
case 2
  idx = m<elt;
case 3
  idx = m>=elt;
case 4
  idx = m<=elt;
end
f = flatten(m(idx));
