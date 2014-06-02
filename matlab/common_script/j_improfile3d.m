function varargout = j_improfile3d(varargin)
%IMPROFILE Volume-value cross-sections along line segments.
% 
%   IMPROFILE computes the intensity values along a line or a multiline path
%   in a volume. IMPROFILE selects equally spaced points along the path you
%   specify, and then uses interpolation to find the intensity value for
%   each point. IMPROFILE works with grayscale intensity. It DOES NOT work
%   with RGB.
%
%	THIS FUNCTION IS A MODIFIED VERSION OF THE improfile.m MATLAB FUNCTION 
%	(IMAGE PROCESSING TOOLBOX).
%	
% INPUT
% img				nxnxn uint8 or double.
% xi				1xm integer. n>1
% yi				1xm integer. n>1
% zi				1xm integer. n>1
% (n)				integer. Number of samples along the segment.
% (method)			'nearest', 'linear'*, 'spline', 'cubic'
%					*:default
% 
% OUTPUT
% c
% (ind)				px3			coordinates of points on the line
% 
%   Example
%   c = j_improfile3d(img,xi,yi,zi)
%
% TODO
%
% Author: Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% Created: 2011-09-18
% Modified: 2011-09-18
% 
% =========================================================================


% load varargin
[xa,ya,za,a,n,method,prof,getn,getprof] = parse_inputs(varargin{:});

% RGB_image = (ndims(a)==3);

% Parametric distance along segments
s = [0;cumsum(sqrt(sum((diff(prof).^2)')'))];

% if segment is very small, exit program
if s(2)-s(1)<0.01
    switch nargout
    case 0,   % If zg was [], we didn't plot and ended up here
        return
    case 1,
        varargout{1} = 0;
	case 2,
        varargout{1} = 0;
		ind = [varargin{2}(1) varargin{3}(1) varargin{3}(1)];
        varargout{2} = ind;
	end
end

% Remove duplicate points if necessary.
killIdx = find(diff(s) == 0);
if (~isempty(killIdx))
    s(killIdx+1) = [];
    prof(killIdx+1,:) = [];
end

ma = size(a,1);
na = size(a,2);
oa = size(a,3);
xmin = min(xa(:)); ymin = min(ya(:)); zmin = min(za(:));
xmax = max(xa(:)); ymax = max(ya(:)); zmax = max(za(:));

if na>1
    dx = max( (xmax-xmin)/(na-1), eps );  
    xxa = xmin:dx:xmax;
else
    dx = 1;
    xxa = xmin;
end

if ma>1
    dy = max( (ymax-ymin)/(ma-1), eps );
    yya = ymin:dy:ymax;
else
    dy = 1;
    yya = ymin;
end

if oa>1
    dz = max( (zmax-zmin)/(oa-1), eps );
    zza = zmin:dz:zmax;
else
    zy = 1;
    zza = zmin;
end

if getn,
    d = abs(diff(prof./(ones(size(prof,1),1)*[dx dy dz])));
    n = max(sum(max(ceil(d)')),3); % In voxel coordinates
end

% Interpolation points along segments
if ~isempty(prof)
    profi = interp1(s,prof,0:(max(s)/(n-1)):max(s));
    xg = profi(:,1);
    yg = profi(:,2);
    zg = profi(:,3);
else
    xg = []; yg = []; zg = [];
end

if ~isempty(a) && ~isempty(xg)
	% Image values along interpolation points - the g stands for Grayscale
	cg = interp3(xxa,yya,zza,a,xg,yg,zg,method);

	% Get profile points in pixel coordinates
	xg_pix = round(axes2pix(na, [xmin xmax], xg)); 
	yg_pix = round(axes2pix(ma, [ymin ymax], yg));  
	zg_pix = round(axes2pix(oa, [zmin zmax], zg));  
    
    % If the result is uint8, Promote to double and put NaN's in the places
    % where the profile went out of the image axes (these are zeros because
    % there is no NaN in UINT8 storage class)
    if ~isa(zg, 'double')     
        prof_hosed = find( (xg_pix<1) | (xg_pix>na) | ...
                           (yg_pix<1) | (yg_pix>ma) );
        if RGB_image
            zr = double(zr); zg = double(zg); zb = double(zb);
            zr(prof_hosed) = NaN;
            zg(prof_hosed) = NaN;
            zb(prof_hosed) = NaN;
        else
            zg = double(zg);
            zg(prof_hosed) = NaN;
        end                 
    end
else
    % empty profile or image data
    % initialize zr/zg/zb for RGB images; just zg for grayscale images;
    [zr zg zb] = deal([]);
end

% if nargout == 0 && ~isempty(zg) % plot it
%     if getprof,
%         h = get(0,'children');
%         fig = 0;
%         for i=1:length(h),
%             if strcmp(get(h(i),'name'),'Profile'),
%                 fig = h(i);
%             end
%         end
%         if ~fig, % Create new window
%             fig = figure('Name','Profile');
%         end
%         figure(fig)
%     else
%         gcf;
%     end
%     if length(prof)>2
%         if RGB_image
%             plot3(xg,yg,zr,'r',xg,yg,zg,'g',xg,yg,zb,'b');
%             set(gca,'ydir','reverse');
%             xlabel X, ylabel Y;
%         else
%             plot3(xg,yg,zg,'b');
%             set(gca,'ydir','reverse');
%             xlabel X, ylabel Y;
%         end
%     else
%         if RGB_image
%             plot(sqrt((xg-xg(1)).^2+(yg-yg(1)).^2),zr,'r',...
%                  sqrt((xg-xg(1)).^2+(yg-yg(1)).^2),zg,'g',...
%                  sqrt((xg-xg(1)).^2+(yg-yg(1)).^2),zb,'b');
%             xlabel('Distance along profile');
%         else
%             plot(sqrt((xg-xg(1)).^2+(yg-yg(1)).^2),zg,'b');
%             xlabel('Distance along profile');
%         end
%     end
% else
%     
%     if RGB_image
%         zg = cat(3,zr(:),zg(:),zb(:));
%     else
%         zg = zg(:);
%     end
%     xi = prof(:,1);
%     yi = prof(:,2);
	
    switch nargout
    case 0,   % If zg was [], we didn't plot and ended up here
        return
    case 1,
        varargout{1} = cg;
	case 2,
        varargout{1} = cg;
		ind = [xg yg zg];
        varargout{2} = ind;
		
			
%     case 3,
%         varargout{1} = xg;
%         varargout{2} = yg;
%         varargout{3} = zg;
%     case 5,
%         varargout{1} = xg;
%         varargout{2} = yg;
%         varargout{3} = zg;
%         varargout{4} = xi;
%         varargout{5} = yi;
%     otherwise
%         msgId = 'Images:improfile:invalidNumOutputArguments';
%         msg = 'Tne number of output arguments is invalid.';
%         error(msgId,'%s',msg);
    end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Function: parse_inputs
%

function [Xa,Ya,Za,Img,N,Method,Prof,GetN,GetProf]=parse_inputs(varargin)
% Outputs:
%     Xa        2 element vector for non-standard axes limits
%     Ya        2 element vector for non-standard axes limits
%     Za        2 element vector for non-standard axes limits
%     A         Image Data
%     N         number of image values along the path (Xi,Yi) to return
%     Method    Interpolation method: 'nearest','bilinear', or 'bicubic'
%     Prof      Profile Indices
%     GetN      Determine number of points from profile if true.
%     GetProf   Get profile from user via mouse if true also get data from image.

% Set defaults
N = [];
GetN = 1;    
GetProf = 0; 
GetCoords = 1;  %     GetCoords - Determine axis coordinates if true.
Method = 'linear';

switch nargin
% case 0,            % improfile
%     GetProf = 1; 
%     GetCoords = 0;
%     
% case 1,            % improfile(n) or improfile('Method')
%     if ischar(varargin{1})
%         Method = varargin{1}; 
%     else 
%         N = varargin{1}; 
%         GetN = 0; 
%     end
%     GetProf = 1; 
%     GetCoords = 0;
%     
% case 2,            % improfile(n,'method')
%     Method = varargin{2};
%     N = varargin{1}; 
%     GetN = 0; 
%     GetProf = 1; 
%     GetCoords = 0;
    
case 4,   % improfile(a,xi,yi,zi)
    A = varargin{1};
    Xi = varargin{2}; 
    Yi = varargin{3}; 
    Zi = varargin{4}; 
    
case 5,   % improfile(a,xi,yi,zi,n) or improfile(a,xi,yi,zi,'method')
    A = varargin{1};
    Xi = varargin{2}; 
    Yi = varargin{3}; 
    Zi = varargin{4}; 
    if ischar(varargin{5})
		% improfile(a,xi,yi,zi,'method')
        Method = varargin{5}; 
	else 
		% improfile(a,xi,yi,zi,n)
        N = varargin{5}; 
        GetN = 0; 
	end
	
case 6,   % improfile(a,xi,yi,zi,n,'method')
    A = varargin{1};
    Xi = varargin{2}; 
    Yi = varargin{3}; 
    Zi = varargin{4}; 
	N = varargin{5}; 
	GetN = 0;
	Method = varargin{6};

% case 5, % improfile(x,y,a,xi,yi) or improfile(a,xi,yi,n,'method')
%     if ischar(varargin{5}), 
%         A = varargin{1};
%         Xi = varargin{2}; 
%         Yi = varargin{3}; 
%         N = varargin{4}; 
%         Method = varargin{5}; 
%         GetN = 0; 
%     else
%         GetCoords = 0;
%         Xa = varargin{1}; 
%         Ya = varargin{2}; 
%         A = varargin{3};
%         Xi = varargin{4}; 
%         Yi = varargin{5}; 
%     end
%     
% case 6, % improfile(x,y,a,xi,yi,n) or improfile(x,y,a,xi,yi,'method')
%     Xa = varargin{1}; 
%     Ya = varargin{2}; 
%     A = varargin{3};
%     Xi = varargin{4}; 
%     Yi = varargin{5}; 
%     if ischar(varargin{6}), 
%         Method = varargin{6}; 
%     else 
%         N = varargin{6};
%         GetN = 0; 
%     end
%     GetCoords = 0;
%     
% case 7, % improfile(x,y,a,xi,yi,n,'method')
%     if ~ischar(varargin{7}) 
%         msgId = 'Images:improfile:invalidInputArrangementOrNumber';
%         msg = 'The arrangement or number of input arguments is invalid.';
%         error(msgId,'%s', msg);
%     end
%     Xa = varargin{1}; 
%     Ya = varargin{2}; 
%     A = varargin{3};
%     Xi = varargin{4}; 
%     Yi = varargin{5}; 
%     N = varargin{6};
%     Method = varargin{7}; 
%     GetN = 0;
%     GetCoords = 0; 
    
otherwise
    msgId = 'Images:improfile:invalidInputArrangementOrNumber';
    msg = 'The arrangement or number of input arguments is invalid.';
    error(msgId, '%s', msg);
end

% set Xa, Ya and Za if unspecified
if (GetCoords && ~GetProf),
    Xa = [1 size(A,1)];
    Ya = [1 size(A,2)];
    Za = [1 size(A,3)];
end

% error checking for N
if (GetN == 0)
    if (N<2 || ~isa(N, 'double'))
        msgId = 'Images:improfile:invalidNumberOfPoints';
        msg = 'N must be a number greater than 1.';
        error(msgId,'%s', msg);
    end
end
% 
% Get profile from user if necessary using data from image
% if GetProf, 
%     [Xa,Ya,A,state] = getimage;
%     if ~state
%         msgId = 'Images:improfile:noImageinAxis';
%         msg = 'Requires an image in the current axis.';
%         error(msgId,'%s',msg);
%     end
%     Prof = getline(gcf); % Get profile from user
% else  % We already have A, Xi, and Yi
    if numel(Xi) ~= numel(Yi)
        msgId = 'Images:improfile:invalidNumberOfPoints';
        msg = 'Xi and Yi must have the same number of points.';
        error(msgId, '%s',msg);
    end
    Prof = [Xi(:) Yi(:) Zi(:)]; % [xi yi zi]
% 
% % error checking for A
% if (~isa(A,'double') && ~isa(A,'uint8') && ~isa(A, 'uint16') && ~islogical(A)) ...
%       && ~isa(A,'single') && ~isa(A,'int16')
%     msgId = 'Images:improfile:invalidImage';
%     msg = 'I must be double, uint8, uint16, int16, single, or logical.';
%     error(msgId, '%s', msg);
% end
% 
% Promote the image to single if it is not logical or if we aren't using nearest.
if islogical(A) || (~isa(A,'double') && ~strcmp(Method,'nearest')) 
    Img = single(A);
else
    Img = A;
end
% 
% % error checking for Xa and  Ya
% if (~isa(Xa,'double') || ~isa(Ya, 'double'))
%     msgId = 'Images:improfile:invalidClassForInput';
%     msg = 'All inputs other than I must be of class double.';
%     error(msgId,'%s',msg);
% end   
% 
% % error checking for Xi and Yi
% if (~GetProf && (~isa(Xi,'double') || ~isa(Yi, 'double')))
%     msgId = 'Images:improfile:invalidClassForInput';
%     msg = 'All inputs other than I must be of class double.';
%     error(msgId,'%s',msg);
% end

%error checking for Method
iptcheckstrs(Method,{'nearest', 'linear', 'spline', 'cubic'}, mfilename,'METHOD',nargin);
