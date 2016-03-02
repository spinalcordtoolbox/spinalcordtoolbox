function f = splitruns(v)

% function f = splitruns(v)
% 
% <v> is a vector of numbers or NaNs
%
% find sequences of non-NaN numbers.  return a cell vector of things
% where each thing is a vector of consecutive indices that refer to the
% non-NaN numbers.
%
% example:
% isequal(splitruns([NaN 5 2 NaN 3 NaN 1 1]),{[2 3] [5] [7 8]})

% calc
vlen = length(v);

% do it
f = {};
cnt = 1;
while 1
  if cnt > vlen
    break;
  end
  if isnan(v(cnt))
    cnt = cnt + 1;
    continue;
  end
  temp = find(isnan(v(cnt:end)));
  if isempty(temp)
    f{end+1} = cnt:vlen;
    break;
  else
    f{end+1} = cnt:(cnt+temp(1)-1)-1;
    cnt = cnt+temp(1);
  end
end
