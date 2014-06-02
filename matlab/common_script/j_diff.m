% =========================================================================
% FUNCTION
%
% derive a 1-d raw matrix
%
% INPUTS
% x             1-d raw matrix (n samples). Function to derivate
% (t)           1-d raw matrix (n samples). Variable used for derivation
%
% OUTPUTS
% xd            1-d raw matrix (n samples)
%
% DEPENDANCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 2006-10-18
% =========================================================================
function varargout = j_diff(x,t)


% initialization
if (nargin<1) help j_diff; return; end
n=length(x);
if (nargin<2) t=(1:1:n); end

% compute first derivative
xd(1)=(x(2)-x(1))/(t(2)-t(1));

% compute next derivatives
for i=2:n-1
    xd(i)=(x(i-1)-x(i+1))/(t(i-1)-t(i+1));
end

% compute last derivative
xd(n)=(x(n)-x(n-1))/(t(n)-t(n-1));

% output
varargout{1} = xd;


