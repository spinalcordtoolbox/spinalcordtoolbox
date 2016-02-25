function options = defaultoptimset(extraopt)

% function options = defaultoptimset(extraopt)
%
% <extraopt> (optional) is a cell vector of parameter/value pairs.
%   default: {}.
%
% return a default optimset options structure (see code) plus anything
% specified by <extraopt>.
%
% example:
% defaultoptimset

% input
if ~exist('extraopt','var') || isempty(extraopt)
  extraopt = {};
end

% do it
options = optimset('Display','iter','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-10,'TolX',1e-10,extraopt{:});
