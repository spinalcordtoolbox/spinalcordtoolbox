% =========================================================================
% FUNCTION
%
% Normalize mean and variance of a signal
%
% INPUTS
% x             (1,n) float.
%
% OUTPUTS
% xn            normalized vector
% x_mean        mean
%
% DEPENDANCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 2006-12-24
% =========================================================================
function varargout = j_normalize(varargin)


% initialization
if (nargin<1) help j_normalize; return; end
x = varargin{1};


% normalize vector
xn = x-mean(x);

if std(xn)
    xn = xn/std(xn);
end

% output
varargout{1} = xn;
varargout{2} = mean(x);
varargout{3} = var(x);
