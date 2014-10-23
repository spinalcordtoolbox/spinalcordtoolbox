function idx = randidx( siz, fraction )
%randidx Selects randomly a fraction of indices for a matrix of size siz

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

n = prod(siz);
idx = randperm(n) ;
idx = idx(1:round(fraction * n));
end

