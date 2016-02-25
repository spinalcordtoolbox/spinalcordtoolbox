function f = calccodcell(x,y,dim)

% function f = calccodcell(x,y,dim)
%
% <x>,<y> are cell vectors of matrices.  the cell vectors should
%   have the same dimensions and their matrix elements should have
%   the same dimensions too.
% <dim> is the dimension of interest
%
% simply return calccod(catcell(dim,x),catcell(dim,y),dim,0,0) but 
% do so in a way that doesn't cause too much memory usage.
% note that NaNs must not exist in <x> nor <y>!
% also, note that this function has less functionality than the 
% original calccod.m.
%
% example:
% a = calccod([1 2 3; 4 5 6],[2 3 4; 5 6 7],1,0,0);
% a2 = calccodcell({[1 2 3] [4 5 6]},{[2 3 4] [5 6 7]},1);
% allzero(a-a2)

numer = cellfun(@(a,b) sum((a-b).^2,dim),x,y,'UniformOutput',0);
denom = cellfun(@(a)   sum(a.^2,dim),      y,'UniformOutput',0);
f = 100*(1 - zerodiv(sum(catcell(dim,numer),dim),sum(catcell(dim,denom),dim),NaN,0));
