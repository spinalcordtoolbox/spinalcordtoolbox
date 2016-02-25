function f = l1vectorlength(m,dim)

% function f = l1vectorlength(m,dim)
%
% <m> is a matrix
% <dim> (optional) is the dimension of interest.
%   if supplied, calculate L1 vector length of each case oriented along <dim>.
%   if [] or not supplied, calculate L1 vector length of entire matrix
%
% calculate L1 vector length of <m>, either of individual cases (in which case
% the output is the same as <m> except collapsed along <dim>) or globally
% (in which case the output is a scalar).
%
% we ignore NaNs gracefully.
% 
% note some weird cases:
%   vectorlength([]) is [].
%   vectorlength([NaN NaN]) is 0
%
% example:
% a = [1 1];
% isequal(l1vectorlength(a),2)
% a = [1 NaN; NaN NaN];
% isequal(l1vectorlength(a,1),[1 0])

% deal with NaNs
m(isnan(m)) = 0;

% handle weird case up front
if isempty(m)
  f = [];
  return;
end

% do it
if ~exist('dim','var') || isempty(dim)
  f = sum(abs(m(:)),1);
else
  f = sum(abs(m),dim);
end
