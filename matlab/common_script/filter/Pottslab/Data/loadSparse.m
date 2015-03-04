function y = loadSparse(type, nSamples)
%LOADSPARSE Creates a sparse signal or image

% written by M. Storath
% $Date: 2013-09-11 13:37:23 +0200 (Mi, 11 Sep 2013) $	$Revision: 79 $

d=0;
if not(exist('nSamples', 'var'))
    nSamples = 2^(8 + d);
end


y = zeros(nSamples, 1);


switch type
    case 'sig1'
        m = 2^d;
        y([50, 170] * m) = 2;
        y([100, 200] * m) = 3;
        y([190] * m) = -4;
        y([90] * m) = -2;
        y([120] * m) = 10;
    
    case 'sig2'
        m = 2^d;
        y([50, 170] * m) = 2;
        y([100, 200] * m) = 6;
        y([190] * m) = -4;
        y([90] * m) = -2;
        y([120] * m) = 10;
        
    case 'sig3'
        m = 2^d;
        y([50, 170] * m) = 2;
        y([100, 200] * m) = 3;
        y([190] * m) = -4;
        y([90] * m) = -2;
        y([120] * m) = 10;
        y([25, 75, 250] * m) = [-1, -1, 1];
      
    case 'sigrand'
        idx = randidx(nSamples, 0.01);
        val = (rand(numel(idx), 1) - 0.5) * 5;
        y(idx) = val;
      
    case 'imgrand'
        n = 30;
        y = zeros(n);
        idx = randidx(numel(y), 0.02);
        %y(idx) = rand(1,numel(idx));
        y(idx) =  ceil(2 * (rand(1,numel(idx) )))/2 ;
        
    case 'img1'
        n = 40;
        y = zeros(n);
        a = [8, 7, 19, 35, 25];
        b = [10, 37, 17, 10, 30]; 
        y(a + n * b) = 1;
        
    case 'img2'
        n = 20;
        y = zeros(n);
        a = [8, 7, 8, 19, 6, 9];
        b = [10, 17, 18, 10, 17, 10]; 
        c = [-1, 2, 3, 1, 2, 2];
        y(a + n * b) = c;
        
    case 'img3'
        n = 20;
        y = zeros(n);
        a = [8, 7, 19];
        b = [10, 17, 10]; 
        y(a + n * b) = 1;
        
    case 'img4'
        n = 100;
        y = zeros(n);
        a = [8, 7, 19, 35, 25, 90, 70, 70];
        b = [10, 37, 17, 10, 30, 60, 30, 15]; 
        y(a + n * b) = 1;
        
    case 'img5'
        n = 30;
        y = zeros(n);
        a = [27, 28, 29, 29, 28, 27];
        b = 23:28; 
        y(a + n * b) = 1;
        a = [20, 20];
        b = [23,28]; 
        y(a + n * b) = -2;
        
    otherwise
        error('Signal does not exist');
        
end

% cast to double
y = double(y);


end
