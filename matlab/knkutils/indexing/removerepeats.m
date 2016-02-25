function v = removerepeats(v,elt)

% function v = removerepeats(v,elt)
%
% <v> is a vector
% <elt> (optional) is a value.  Default: NaN.
%
% given a vector <v>, replace any runs of the same item with just the
% first instance, replacing the remaining instances with <elt>.
% we use == to check for equality of items.
%
% example:
% removerepeats([1 1 2 3 6 6 6 1 1])

% inputs
if ~exist('elt','var') || isempty(elt)
  elt = NaN;
end

% do it
p = 1;
while p <= length(v)
  temp = v(p+1:end) == v(p);
  temp2 = find(temp==0);
  if isempty(temp2)
    v(p+1:end) = elt;
    break;
  end
  v(p+1:p+temp2(1)-1) = elt;
  p = p+temp2(1);
end
