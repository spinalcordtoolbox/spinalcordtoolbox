function vect = SmartVec(start,slength,step,mode)
% vect = SmartVec(start,slength,step,mode)
% makes sequence that satisfies certain conditions.
% 
% START is a scalar or vector with starting values of an sequence
% SLENGTH is a scalar or vector (with the same length as START) and
%   indicates the length of all sequences or the length of the
%   corresponding sequence in START.
% STEP is optional and indicates the stepsize of the series. Default is 1
%   STEP can be a scalar and indicate the stepsize for all segments, or it
%   can be the same lenght as START if you want to provide a different
%   stepsize for each segment
% MODE is optional and has two options:
%   'neg'  : the produced sequence will satisfy  : diff([n n+1]) = -1 
%   'flat' : the produced sequence will satisfy  : diff([n n+1]) =  0
% if no mode, the produced sequence will satisfy : diff([n n+1]) =  1
% MODE can be the third input to the function if you do not wish to
% override the default stepsize
%
% examples:
%   SmartVec([1 5],3)
%   ans =
%        1     2     3     5     6     7
%   SmartVec([1 5],[3 5])
%   ans =
%        1     2     3     5     6     7     8     9
%   SmartVec([1 5],[3 5],'neg')
%   ans =
%        1     0    -1     5     4     3     2     1
%   SmartVec([1 5],[3 5],2,'neg')
%   ans =
%        1     -1   -3     5     3     1     -1    -3
%   SmartVec([1 5],3,'neg')
%   ans =
%        1     0    -1     5     4     3
%   SmartVec([1 5],[3 5],'flat')
%   ans =
%        1     1     1     5     5     5     5     5
%   SmartVec([1 5],3,'flat')
%   ans =
%        1     1     1     5     5     5

% DN 2008-04-21 Written it, extention of SmartVec
% DN 2008-09-05 Updated engine, no more meshgrid, less transposes needed
% DN 2010-08-01 Now supports different stepsize than 1 for vectors
% DN 2011-07-23 Now supports different stepsizes for each segment

% check input
psychassert(length(slength)==1 || length(slength)==length(start),'slength must be a scalar or a vector with the same length as start');
qplat   = false;
qneg    = false;
mult    = 1;
if nargin >= 3
    qhavemode = false;
    % backwards compatibility, mode can be third input as well
    if nargin==3 && ischar(step)
        mode = step;
        qhavemode = true;
    end
    if nargin==4 || qhavemode
        switch mode
            case 'neg'
                qneg    = true;
            case 'flat'
                qplat   = true;
            otherwise
                error('Mode "%s" not recognized.\nUse "neg" of "flat".',mode)
        end
    end
    if isnumeric(step) && ~isempty(step)
        mult = step;
    end
end
psychassert(length(mult)==1 || length(mult)==length(start),'step must be a scalar or a vector with the same length as start');
maxlen      = max(slength);
slength     = slength(:);

% do the work
start       = start(:)';
startmat    = start(ones(maxlen,1),:);

if ~qplat
    vec         = [0:maxlen-1]';
    vecmat      = vec(:,ones(1,length(start)));
    if any(mult~=1)
        if ~isscalar(mult)
            vecmat  = bsxfun(@times,mult,vecmat);
        else
            vecmat  = mult .* vecmat;
        end
    end
    if qneg
        vecmat  = -1*vecmat;
    end
else
    vecmat      = zeros(maxlen,length(start));
end
    
totmat      = startmat + vecmat;
vect        = totmat(:)';

% if slength is a vector, we have to trim parts of the output
if ~isscalar(slength) && length(unique(slength))>1
    MinInd  = cumsum([0 maxlen*ones(1,length(slength)-1)]).'+1;
    MaxInd  = num2cell(MinInd+slength-1);
    indcell = cellfun(@(a,b)a:b,num2cell(MinInd),MaxInd,'UniformOutput',false);
    vect    = vect([indcell{:}]);
end
