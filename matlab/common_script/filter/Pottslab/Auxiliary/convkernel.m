function h = convkernel( type, n, scale )
%CONVKERNEL Creates a convolution kernel

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

if not(exist('scale', 'var'))
   scale = 1; 
end

% set coordinates
d = n/2;
x = linspace(-d, d, n)/scale;

switch type
    case 'box'
        h = ones(n, 1);
        
    case 'hat'
        h(x >= 0) = 1-x(x >= 0);
        h(x < 0) = 1 + x(x < 0);
        h(abs(x) > 1) = 0; 
        
    case 'gaussian'
        h = exp(-0.5 * x.^2);
        
    case 'doubleexp'
        h = exp(-0.5 * abs(x));
        
    case 'mollifier'
        h = exp(-1./(1 - abs(x).^2));
        h(abs(x) > 1) = 0;
        
    case 'ramp'
        h(x >= 0) = 1-x(x >= 0);
        h(abs(x) > 1) = 0; 
        
    otherwise
        error('This option does not exist.')
end

% normalize
h = h(:)./sum(h);

end

