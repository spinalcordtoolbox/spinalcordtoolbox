function f = normalizerange(m,targetmin,targetmax,sourcemin,sourcemax,chop,mode,fast)

% function f = normalizerange(m,targetmin,targetmax,sourcemin,sourcemax,chop,mode,fast)
%
% <m> is a matrix
% <targetmin> is the minimum desired value.  can be a scalar or a matrix the same size as <m>.
% <targetmax> is the maximum desired value.  can be a scalar or a matrix the same size as <m>.
% <sourcemin> (optional) sets the min value of <m>.  can be a scalar or a matrix the same size as <m>.
%   default is [], which means to find the actual minimum.  special case is NaN which means -nanmax(abs(m(:))).
% <sourcemax> (optional) sets the max value of <m>.  can be a scalar or a matrix the same size as <m>.
%   default is [], which means to find the actual maximum.  special case is NaN which means nanmax(abs(m(:))).
% <chop> (optional) is whether to chop off the ends such that there are no values below <targetmin> nor
%   above <targetmax>.  default: 1.
% <mode> (optional) is
%   0 means normal operation
%   1 means interpret <sourcemin> and <sourcemax> as multipliers for the std of m(:).
%     in this mode, the default for <sourcemin> and <sourcemax> is -3 and 3, respectively,
%     which means to use mn-3*sd and mn+3*sd for the min and max value of <m>, respectively.
%     note that in this mode, <sourcemin> and <sourcemax> cannot be NaN.
%   default: 0.
% <fast> (optional) means we have a guarantee that all inputs are fully specified and <m> is not empty.
%
% return <m> scaled and translated such that [<sourcemin>,<sourcemax>] maps to
% [<targetmin>,<targetmax>].  if <chop>, we also threshold values below <targetmin>
% and values above <targetmax>.
%
% note that if <sourcemin> is ever equal to <sourcemax>, then we die with an error.
% note that <chop> has no effect if <sourcemin> and <sourcemax> aren't specified.
%
% we deal with NaNs in <m> gracefully.
%
% examples:
% isequal(normalizerange([1 2 3],0,1),[0 1/2 1])
% isequal(normalizerange([1 2 3],0,1,2,3,1),[0 0 1])
% isequalwithequalnans(normalizerange([1 2 NaN],0,1,0,4),[1/4 2/4 NaN])

% if <fast>, skip stuff for speed
if nargin ~= 8

  % check empty case
  if isempty(m)
    f = m;
    return;
  end
  
  % input
  if ~exist('sourcemin','var') || isempty(sourcemin)
    sourcemin = [];
  end
  if ~exist('sourcemax','var') || isempty(sourcemax)
    sourcemax = [];
  end
  if ~exist('chop','var') || isempty(chop)
    chop = 1;
  end
  if ~exist('mode','var') || isempty(mode)
    mode = 0;
  end

end

% calc
skipchop = (mode==0 && (isempty(sourcemin) && isempty(sourcemax))) || (mode==0 && isnan(sourcemin) && isnan(sourcemax));  % don't bother chopping in these cases
switch mode
case 0
  if isempty(sourcemin)
    sourcemin = nanmin(m(:));
  end
  if isempty(sourcemax)
    sourcemax = nanmax(m(:));
  end
  if isnan(sourcemin) || isnan(sourcemax)
    temp = nanmax(abs(m(:)));
    if isnan(sourcemin)
      sourcemin = -temp;
    end
    if isnan(sourcemax)
      sourcemax = temp;
    end
  end
case 1
  if isempty(sourcemin)
    sourcemin = -3;
  end
  if isempty(sourcemax)
    sourcemax = 3;
  end
  mn = nanmean(m(:));
  sd = nanstd(m(:));
  sourcemin = mn+sourcemin*sd;
  sourcemax = mn+sourcemax*sd;
end

% sanity check
if any(sourcemin==sourcemax)
  error('sourcemin and sourcemax are the same in at least one case');
end

% go ahead and chop
if chop && ~skipchop
  temp = isnan(m);
  m = max(min(m,sourcemax),sourcemin);
  m(temp) = NaN;  % preserve NaNs
end

% want to do: f = (m-sourcemin) .* (targetmax-targetmin)./(sourcemax-sourcemin) + targetmin
val = (targetmax-targetmin)./(sourcemax-sourcemin);
f = m.*val - (sourcemin.*val - targetmin);  % like this for speed
