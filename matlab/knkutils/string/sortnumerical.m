function [f,ix] = sortnumerical(m,mode)

% function [f,ix] = sortnumerical(m)
%
% <m> is a cell vector of strings
% 
% return <m> in sorted order.  this is done by extracting 
% (\d+)\D*$ from each element of <m> and then sorting based 
% on those numbers (if they exist) to break ties.  we also return
% the vector of indices that we used to sort with.
%
% this function is useful for when you name files with %d instead of 
% something like %04d.
%
% example:
% sortnumerical({'test1' 'test10' 'test2' 'ok1' 'ok20'})

% input
if ~exist('mode','var') || isempty(mode)
  mode = 2;
end

% this mode is to sort solely based on the numbers
switch mode
case 0

  % extract the numbers
  rec = [];
  for p=1:length(m)
    temp = regexp(m{p},'(\d+)\D*$','tokens');
    if length(temp)==1
      rec(p) = str2double(temp{1}{1});
    else
      rec(p) = 0;
    end
  end
  
  % sort them
  [s,ix] = sort(rec);
  f = m(ix);

% this mode is to sort based on everything until the numbers
case 1

  % extract the numbers
  rec = {};
  for p=1:length(m)
    temp = regexp(m{p},'^(.*?)\d+\D*$','tokens');
    if length(temp)==1
      rec{p} = temp{1}{1};
    else
      rec{p} = m{p};
    end
  end
  
  % sort them
  [s,ix] = sort(rec);
  f = m(ix);

% the normal case
case 2

  [d,ix] = sortnumerical(m,0);
  [d,ix2] = sortnumerical(m(ix),1);
  f = m(ix(ix2));
  ix = ix(ix2);

end
