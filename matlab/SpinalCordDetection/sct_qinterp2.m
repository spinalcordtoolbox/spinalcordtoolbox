function Zi = qinterp2(X,Y,Z,xi,yi,methodflag)
%QINTERP2 2-dimensional fast interpolation
% qinterp2 provides a speedup over interp2 in the same way that
% qinterp1 provides a speedup over interp1
%
% Usage:
%   yi = qinterp2(X,Y,Z,xi,yi)  - Same usage as interp2, X and Y should be
%                                 "plaid" (e.g., generated with meshgrid).
%                                 Defaults to bilinear interpolation
%   yi = qinterp2(...,flag)
%           flag = 0       - Nearest-neighbor
%           flag = 1       - Triangular-mesh linear interpolation.
%           flag = 2       - Bilinear (equivalent to MATLAB's 'linear')
%
% Usage restrictions
%   X(:,n) and Y(m,:) must be monotonically and evenly increasing
%   e.g.,  [X,Y] = meshgrid(-5:5,0:0.025:1);
%
% Examples:
%   % Set up the library data for each example
%   [X,Y] = meshgrid(-4:0.1:4,-4:0.1:4);
%   Z = exp(-X.^2-Y.^2);
%
%   % Interpolate a line
%   xi = -4:0.03:4; yi = xi;
%   Zi = qinterp2(X,Y,Z,xi,yi);
%   % Plot the interpolant over the library data
%   figure, mesh(X,Y,Z), hold on, plot3(xi,yi,Zi,'-r');
%
%   % Interpolate a region
%   [xi,yi] = meshgrid(-3:0.3:0,0:0.3:3);
%   Zi = qinterp2(X,Y,Z,xi,yi);
%   % Plot the interpolant
%   figure, mesh(X,Y,Zi);
%
% Error checking
%   WARNING: Little error checking is performed on the X or Y arrays. If these
%   are not proper monotonic, evenly increasing plaid arrays, this
%   function will produce incorrect output without generating an error.
%   This is done because error checking of the "library" arrays takes O(mn)
%   time (where the arrays are size [m,n]).  This function is
%   algorithmically independent of the size of the library arrays, and its
%   run time is determine solely by the size of xi and yi
%
% Using with non-evenly spaced arrays:
%   See qinterp1

% Search array error checking
if size(xi)~=size(yi)
    error('%s and %s must be equal size',inputname(4),inputname(5));
end

% Library array error checking (size only)
if size(X)~=size(Y)
    error('%s and %s must have the same size',inputname(1),inputname(2));
end
librarySize = size(X);

%{
% Library checking - makes code super slow for large X and Y
DIFF_TOL = 1e-14;
if ~all(all( abs(diff(diff(X'))) < DIFF_TOL*max(max(abs(X))) ))
    error('%s is not evenly spaced',inputname(1));
end
if ~all(all( abs(diff(diff(Y)))  < DIFF_TOL*max(max(abs(Y))) ))
    error('%s is not evenly spaced',inputname(2));
end
%}

% Decide the interpolation method
if nargin>=6
    method = methodflag;
else
    method = 2; % Default to bilinear
end

% Get X and Y library array spacing
ndx = 1/(X(1,2)-X(1,1));    ndy = 1/(Y(2,1)-Y(1,1));
% Begin mapping xi and yi vectors onto index space by subtracting library
% array minima and scaling to index spacing
xi = (xi - X(1,1))*ndx;       yi = (yi - Y(1,1))*ndy;

% Fill Zi with NaNs
% Zi = NaN*ones(size(xi));
Zi = zeros(size(xi));
switch method
    
    % Nearest-neighbor method
    case 0
        % Find the nearest point in index space
        rxi = round(xi)+1;  ryi = round(yi)+1;
        % Find points that are in X,Y range
        flag = rxi>0 & rxi<=librarySize(2) & ~isnan(rxi) &...
            ryi>0 & ryi<=librarySize(1) & ~isnan(ryi);
        % Map subscripts to indices
        ind = ryi + librarySize(1)*(rxi-1);
        Zi(flag) = Z(ind(flag));
        
    % Linear method
    case 1
        % Split the square bounded by (x_i,y_i) & (x_i+1,y_i+1) into two
        % triangles.  The interpolation is given by finding the function plane
        % defined by the three triangle vertices
        fxi = floor(xi)+1;  fyi = floor(yi)+1;   % x_i and y_i
        dfxi = xi-fxi+1;    dfyi = yi-fyi+1;     % Location in unit square
        
        ind1 = fyi + librarySize(1)*(fxi-1);     % Indices of (  x_i  ,  y_i  )
        ind2 = fyi + librarySize(1)*fxi;         % Indices of ( x_i+1 ,  y_i  )
        ind3 = fyi + 1 + librarySize(1)*fxi;     % Indices of ( x_i+1 , y_i+1 )
        ind4 = fyi + 1 + librarySize(1)*(fxi-1); % Indices of (  x_i  , y_i+1 )
        
        % flagIn determines whether the requested location is inside of the
        % library arrays
        flagIn = fxi>0 & fxi<librarySize(2) & ~isnan(fxi) &...
            fyi>0 & fyi<librarySize(1) & ~isnan(fyi);
        
        % flagCompare determines which triangle the requested location is in
        flagCompare = dfxi>=dfyi;
        
        % This is the interpolation value in the x>=y triangle
        % Note that the equation
        %  A. Is linear, and
        %  B. Returns the correct value at the three boundary points
        % Therefore it describes a plane passing through all three points!
        %
        % From http://osl.iu.edu/~tveldhui/papers/MAScThesis/node33.html
        flag1 = flagIn & flagCompare;
        Zi(flag1) = ...
            Z(ind1(flag1)).*(1-dfxi(flag1)) +...
            Z(ind2(flag1)).*(dfxi(flag1)-dfyi(flag1)) +...
            Z(ind3(flag1)).*dfyi(flag1);
        
        % And the y>x triangle
        flag2 = flagIn & ~flagCompare;
        Zi(flag2) = ...
            Z(ind1(flag2)).*(1-dfyi(flag2)) +...
            Z(ind4(flag2)).*(dfyi(flag2)-dfxi(flag2)) +...
            Z(ind3(flag2)).*dfxi(flag2);

    case 2 % Bilinear interpolation
        % Code is cloned from above for speed
        % Transform to unit square
        fxi = floor(xi)+1;  fyi = floor(yi)+1;   % x_i and y_i
        dfxi = xi-fxi+1;    dfyi = yi-fyi+1;     % Location in unit square
        
        % flagIn determines whether the requested location is inside of the
        % library arrays
        flagIn = fxi>0 & fxi<librarySize(2) & ~isnan(fxi) &...
            fyi>0 & fyi<librarySize(1) & ~isnan(fyi);
        
        % Toss all out-of-bounds variables now to save time
        fxi = fxi(flagIn); fyi = fyi(flagIn);
        dfxi = dfxi(flagIn); dfyi = dfyi(flagIn);
        
        % Find bounding vertices
        ind1 = fyi + librarySize(1)*(fxi-1);     % Indices of (  x_i  ,  y_i  )
        ind2 = fyi + librarySize(1)*fxi;         % Indices of ( x_i+1 ,  y_i  )
        ind3 = fyi + 1 + librarySize(1)*fxi;     % Indices of ( x_i+1 , y_i+1 )
        ind4 = fyi + 1 + librarySize(1)*(fxi-1); % Indices of (  x_i  , y_i+1 )
        
        % Bilinear interpolation.  See
        % http://en.wikipedia.org/wiki/Bilinear_interpolation
        Zi(flagIn) = Z(ind1).*(1-dfxi).*(1-dfyi) + ...
            Z(ind2).*dfxi.*(1-dfyi) + ...
            Z(ind4).*(1-dfxi).*dfyi + ...
            Z(ind3).*dfxi.*dfyi;
        
    otherwise
        error('Invalid method flag');
        
end %switch