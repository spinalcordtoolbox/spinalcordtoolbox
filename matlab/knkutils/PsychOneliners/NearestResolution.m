function res=NearestResolution(screenNumber,width,height,hz,pixelSize,outputId)
% res=NearestResolution(screenNumber,[width,height,hz,pixelSize])
% res=NearestResolution(screenNumber [,width][,height][,hz][,pixelSize][,outputId])
% res=NearestResolution(screenNumber,desiredRes)
%
% Finds the available screen resolution most similar (in log cartesian space) to that
% requested. Any argument that is [] or NaN will be ignored in assessing similarity.
% If you specify pixelSize then the returned res will specify pixelSize. Typically
% you'll use the returned "res" to set your screen's resolution:
%
% SetResolution(screenNumber, res)
%
% Note: On Linux, if more than one video display output is connected to
% a given screenNumber, you must specify the optional 'outputId' parameter
% to select the output for which you want a match.
%
% Also see SetResolution, ResolutionTest, and Screen Resolution and Resolutions.

% HISTORY:
% 1/28/00 dgp Wrote it.
% 9/17/01 dgp Frans Cornelissen, f.w.cornelissen@med.rug.nl, discovered that iBooks
%			  always report a framerate of NaN. So we ignore framerate when
%			  Screen Resolutions returns NaN Hz.
% 4/29/02 dgp Screen Resolutions is only available on Mac.
% 6/6/02  dgp Accept res instead of parameter list.
% 9/23/07 mk  Adapted for Psychtoolbox-3.
% 2/28/08 mk  Fix parsing, error handling and documentation...
% 12/28/12 mk Fix bug reported by St�phane Rainville on GitHub (issue #86).
% 04/15/14 mk Allow selection of outputId for which to match.

if nargin<2 || nargin>6
    error(sprintf('%s\n%s','USAGE: res=NearestResolution(screenNumber,[width,height,hz,pixelSize])',...
        '           res=NearestResolution(screenNumber [,width][,height][,hz][,pixelSize])',...
        '           res=NearestResolution(screenNumber,desiredRes)')); %#ok<*SPERR>
end

if nargin<6
    outputId = [];
end

if nargin<5
    pixelSize=[];
end

if nargin<4
    hz=[];
end

if nargin<3
    height=[];
end

if nargin==2
    % 'desiredRes' could be a struct or a vector:
    if isa(width,'struct')
        % It is a struct:
        res=width;
        
        if isfield(res,'width')
            width=res.width;
        else
            width=[];
        end
        
        if isfield(res,'height')
            height=res.height;
        else
            height = [];
        end
        
        if isfield(res,'hz')
            hz=res.hz;
        else
            hz = [];
        end
        
        if isfield(res,'pixelSize')
            pixelSize=res.pixelSize;
        else
            pixelSize = [];
        end
    else
        % It is a vector:
        res=width;
        
        if ~isempty(res)
            width = res(1);
        else
            error('The second argument "desiredRes" is neither a valid struct, nor a valid vector!');
        end
        
        if length(res)>1
            height = res(2);
        else
            height = [];
        end
        
        if length(res)>2
            hz = res(3);
        else
            hz = [];
        end
        
        if length(res)>3
            pixelSize = res(4);
        else
            pixelSize = [];
        end
    end
end

if ~isscalar(screenNumber) || ~ismember(screenNumber, Screen('Screens'))
    error(sprintf('Invalid screenNumber %d.',screenNumber));
end

% Get a list of available resolutions on output outputId:
res=Screen('Resolutions', screenNumber, outputId);

wish.width=width;
wish.height=height;
wish.hz=hz;
wish.pixelSize=pixelSize;

for i=1:length(res)
    d(i)=distance(wish,res(i)); %#ok<AGROW>
end

[x,i]=min(d); %#ok<ASGLU>
res=res(i);

if isempty(wish.pixelSize) || ~isfinite(wish.pixelSize)
    res.pixelSize = [];
end

return

function d=distance(a,b)
% "a" may omit values, for which you "don't care".
% "a" has "pixelSize" field, but "b" has "pixelSizes" field.
d=0;

if ~isempty(a.width) && isfinite(a.width)
    d=d+log10(a.width/b.width)^2;
end

if ~isempty(a.height) && isfinite(a.height)
    d=d+log10(a.height/b.height)^2;
end

% b.hz may be reported as zero on some operating systems when display is a
% flat-panel. We need to ignore the 'hz' wish in that case, as there will
% be only one fixed framerate on theses systems anyway (usually 60 hz).
if ~isempty(a.hz) && isfinite(a.hz) && ~isempty(b.hz) && isfinite(b.hz) && (b.hz > 0)
    d=d+log10(a.hz/b.hz)^2;
end

return
