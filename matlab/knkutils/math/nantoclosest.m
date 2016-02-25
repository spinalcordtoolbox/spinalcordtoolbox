function v = nantoclosest(v)

% function v = nantoclosest(v)
%
% <v> is a (row or column) vector potentially with NaNs
%
% replace each NaN with the closest non-NaN value.
% if there are ties, we take the mean of the closest
% non-NaN values.
%
% TODO: should we make this work on matrices and faster?
%
% example:
% nantoclosest([1 2 NaN 3 4 5 NaN NaN])

bad = find(isnan(v));
good = find(~isnan(v));
for p=1:length(bad)
  temp = abs(bad(p) - good);
  [mn,ix] = min(temp);
  pos = find(temp==mn);
  v(bad(p)) = mean(v(good(pos)));
end
