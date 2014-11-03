function y = loadPcwConst(type, nSamples)
%LOADPCWCONST Creates a piecewise constant signal or image

% written by M. Storath
% $Date: 2014-05-07 11:38:52 +0200 (Mi, 07 Mai 2014) $	$Revision: 88 $

if not(exist('nSamples', 'var'))
    nSamples = 2^8;
end


x = linspace(-1, 1, nSamples)';
y = zeros(size(x));

switch type
    case 'rect'
        y = (-0.5 < x) & ( x < 0.5);
    case {'step', 'heaviside'}
        y = x >= 0;
    case 'jumps'
        % two jumps
        y = (-0.3 < x) & ( x < -0.1) + 2 * ((0.1 < x) & ( x < 0.4));
    case 'equidistant'
        n = 8; % number of partitions
        step = nSamples / n;
        heights = [3, 1, 7, 6, 5, 0,6,4];
        for i = 1:n
            iv = ((i-1) * step + 1) : (i * step);
            y(iv) = heights(i) / max(heights);
        end
        
    case 'sampleDec'
        heights = [3, 1, 7, 6, 5, 0,6,6];
        jumps = diff(heights);
        jumpDist = [30, 40, 32, 32, 32, 50, 60];
        jumpIdx = cumsum(jumpDist);
        y(jumpIdx) = jumps;
        y = cumsum(y);
        y = mat2gray(y);
        
    case 'sampleDec2'
        heights = [3, 1, 7, 6, 5, 0,6,6];
        jumps = diff(heights);
        jumpDist = [30, 40, 32, 32, 32, 50, 40];
        jumpIdx = cumsum(jumpDist);
        y(jumpIdx) = jumps;
        y = cumsum(y);
        y = mat2gray(y);
        
    case 'equidistant2'
        n = 8; % number of partitions
        step = nSamples / n;
        heights = [3, 1, 7, 6, 5, 4,6,4];
        for i = 1:n
            iv = ((i-1) * step + 1) : (i * step);
            y(iv) = heights(i) / max(heights);
        end
        
    case 'sample1'
        steps = [2, 4, 3, 3, 2, 5, 3, 3, 3];
        steps = (cumsum(steps ./ sum(steps)) - 0.5) * 2;
        heights = [3, 1, 7, 5.5, 4.5, 3.5,6,0];
        n= numel(heights);
        for i = 1:n
            idx = find((steps(i) < x) & (x <= steps(i+1)));
            y( idx ) = heights(i) / max(heights);
        end
        
    case 'sample2'
        steps = [2, 4, 3, 3, 2, 5, 2, 3, 3,6,4,2,8,3,4, 4]; %15
        steps = (cumsum(steps ./ sum(steps)) - 0.5) * 2;
        heights = [3, 1, 7, 6, 5, 4,6,0, 2, 4, 3, 1.5, 7, 9, 2];
        n= numel(heights);
        for i = 1:n
            idx = find((steps(i) < x) & (x <= steps(i+1)));
            y( idx ) = heights(i) / max(heights);
        end
        
   case 'sample3'
        steps = [2, 4, 3, 6, 4, 5, 4, 3, 3];
        steps = (cumsum(steps ./ sum(steps)) - 0.5) * 2;
        heights = [3, 1, 7, 6, 5, 4,6,0];
        n= numel(heights);
        for i = 1:n
            idx = find((steps(i) < x) & (x <= steps(i+1)));
            y( idx ) = heights(i) / max(heights);
        end
        
   case 'sample4'
        %steps = [3, 2, 5, 10, 4, 3, 3];
        %steps = [ -Inf, (cumsum(steps ./ sum(steps)) - 0.5) * 2];
        %heights = [4, 2,3, 3.1, 3.05, 1, 1];
        steps = [2, 4, 2, 2, 1, 1];
        steps = [ -Inf, (cumsum(steps ./ sum(steps)) - 0.5) * 2];
        heights = [0.0, 0.1, 0.05, 1, 0.5, 0];
        n= numel(heights);
        for i = 1:n
            idx = find((steps(i) < x) & (x <= steps(i+1)));
            y( idx ) = heights(i) / max(abs(heights));
        end
        
    case 'geo1'
        y = double(imread('geo1.png'))/255;
        
    case 'geo2'
        y = double(imread('geo2.png'))/255;
        
    case 'geo3'
        y = double(imread('geo3.png'))/255;
        
    case 'geo4'
        y = double(imread('geo4.png'))/255;
        
    case 'geo5'
        y = double(imread('geo5.png'))/255;
        
    case 'geo6'
        y = double(imread('geo6.png'))/255;
        
    case 'geo7'
        y = double(imread('geo7.png'))/255;
        
    case 'geo8'
        y = double(imread('geo8.png'))/255;
        
    case 'overlay'
        I1 = double(imread('rectangle.png'));
        I2 = double(imread('octo.png'));
        y =  (I1 + 2*I2)/ (3*255);
        
    otherwise
        error('This option does not exist.')
end

% cast to double
y = double(y);




end
