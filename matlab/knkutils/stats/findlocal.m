function f = findlocal(x,ref,dist)

% function f = findlocal(x,ref,dist)
%
% <x> is a vector with the coordinates of a point, [x1 x2 x3 ... xN]
% <ref> is N x P with a reference point in each column
% <dist> is a 1 x N vector with non-negative numbers.
%   can be a scalar in which case we assume that value for all cases.
%
% find all <ref> points that are within <dist> of <x>, considered
% along each dimension independently (think: find all points within
% a hyperrectangle that is centered on <x>).  return a vector of 
% (sorted) indices into the second dimension of <ref>.
% 
% note that we make an specific effort towards fast execution!
%
% example:
% [xx,yy] = ndgrid(1:20,1:20);
% ix = findlocal([12 15],[flatten(xx); flatten(yy)],3.5);
% figure; imagesc(copymatrix(zeros(20,20),ix,1));

% input
if length(dist)==1
  dist = repmat(dist,[1 length(x)]);
end

% calculate good indices sequentially
good = {};
for p=1:length(x)
  if p==1
    good{p} = find(abs(x(p) - ref(p,:)) <= dist(p));
  else
    for q=p-1:-1:1
      if q==p-1
        ok = good{q};
      else
        ok = good{q}(ok);
      end
    end
    good{p} = find(abs(x(p) - ref(p,ok)) <= dist(p));
  end
end

% figure out final indices
for q=length(x):-1:1
  if q==length(x)
    ok = good{q};
  else
    ok = good{q}(ok);
  end
end
f = ok;
