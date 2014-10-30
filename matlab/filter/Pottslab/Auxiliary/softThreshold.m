function y = softThreshold(x, tau)
%softThreshold The classical soft threshold function

% written by M. Storath
% $Date: 2013-01-05 17:25:45 +0100 (Sat, 05 Jan 2013) $	$Revision: 63 $

y = (abs(x) - tau) .* (abs(x) > tau) .* sign(x);
end