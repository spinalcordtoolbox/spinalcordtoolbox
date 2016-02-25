function m = catcell(dim,m)

% function m = catcell(dim,m)
%
% <dim> is the dimension to concatenate along
% <m> is a cell matrix
%
% simply return cat(dim,m{:}).  this function is useful because 
% MATLAB doesn't provide an easy way to apply "{:}" to an 
% arbitrary matrix.
%
% example:
% isequal(catcell(2,{1 2 3}),[1 2 3])

m = cat(dim,m{:});



% THIS SEEMED TO FAIL AS A WAY OF SAVING MEMORY
% f = [];
% for p=1:numel(m)
%   if p == 1
%     f = m{p};
%   else
%     f = cat(dim,f,m{p});
%   end
%   m{p} = [];
% end
