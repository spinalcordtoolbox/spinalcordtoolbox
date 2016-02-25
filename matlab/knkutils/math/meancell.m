function f = meancell(m,dim)

% function f = meancell(m,dim)
%
% <m> is a cell matrix
% <dim> is the dimension along which to calculate the mean
%
% simply return mean(catcell(dim,m),dim) but do so in a way
% that doesn't cause too much memory usage.
%
% example:
% meancell({[1 2 3] [4 5] [6]},2)

totalsum = sum(catcell(dim,cellfun(@(x) sum(x,dim),  m,'UniformOutput',0)),dim);
totalcnt = sum(catcell(dim,cellfun(@(x) size(x,dim), m,'UniformOutput',0)),dim);
f = totalsum ./ totalcnt;
