function u = minL2Potts(f, gamma, weights)
%MINL2POTTS Minimizes the classical L^2-Potts problem
%
%Description
% Computes an exact minimizer of the L^2-Potts functional
%
%  \gamma \| D u \|_0 + \| u - f \|_2^2
%
% in O(n^2) time.
%
%Reference:
% F. Friedrich, A. Kempe, V. Liebscher, G. Winkler,
% Complexity penalized M-estimation: Fast computation
% Journal of Computational and Graphical Statistics, 2008
%
% See also: minL2iPotts, minL1Potts, minL1iPotts

% written by M. Storath
% $Date: 2013-10-04 13:43:18 +0200 (Fr, 04 Okt 2013) $	$Revision: 80 $

% weighted and vector valued version exists only in Java
if ~exist('weights', 'var')
    weights = ones(size(f));
end

siz = size(f);

complexData = ~isreal(f);
% convert complex data to vector
if complexData
    f = [real(f(:)), imag(f(:))];
end

u = pottslab.JavaTools.minL2Potts( f, gamma, weights);

% convert from vector to complex
if complexData
    u = u(:,1) + 1i * u(:,2);
end

% reshape u to size of f
u = reshape(u, siz);

%--------------------------------------------------------------------------
%%show the result, if requested
if nargout == 0
    plot(f, '.', 'MarkerSize', 10);
    hold on
    stairs(u, 'r', 'LineWidth', 2);
    hold off
    legend({'Signal', 'L^2-Potts estimate'});
    grid on;
end

end
