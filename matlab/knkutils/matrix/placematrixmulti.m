function f = placematrixmulti(m1,m2,n)

% function f = placematrixmulti(m1,m2,n)
%
% <m1> is a 2D matrix, with potentially some extra stuff in the third dimension.
% <m2> is a 2D matrix, with potentially some extra stuff in the third dimension.
%   note that size(m1,3) must equal size(m2,3).
% <n> is the number of times we should place matrix <m2>
%
% randomly choose a position to place matrix <m2> into <m1>.
% given that position, add <m2> in.  repeat this <n> times.
% note that the range over which we choose positions is exactly
% the range of positions that <m2> overlaps <m1> with any finite
% amount of overlap.
%
% example:
% figure; imagesc(placematrixmulti(zeros(100,100),[1 0; 0 1],100));

% calc
rstart = 1-size(m2,1)+1;
rend = size(m1,1);
cstart = 1-size(m2,2)+1;
cend = size(m1,2);
zz = zeros(size(m1));

% do it
f = m1;
for p=1:n
%  f = f + placematrix(zz,m2,[randint(1,1,[rstart rend]) randint(1,1,[cstart cend])]);
  f = f + placematrix(zz,m2,[floor(rand*(rend-rstart+1))+rstart floor(rand*(cend-cstart+1))+cstart]);
end
