function stop = outputfcnsanitycheck(params,optimValues,state,tol,numsteps)

% function stop = outputfcnsanitycheck(params,optimValues,state,tol,numsteps)
%
% <params> is the current optimization parameter
% <optimValues>,<state> are optimization state stuff
% <tol> (optional) is a positive number.  default: 1e-6.
% <numsteps> (optional) is the positive number of iterations in the 
%   past to compare to.  default: 10.
%
% we look back <numsteps> iterations and check whether the resnorm field
% of <optimValues> has changed by less than <tol>.  if so, we stop the
% optimization.  the only reason for this is to patch up some weird cases
% where the regular optimizer doesn't stop when it should (in these cases,
% the optimizer goes on forever without anything actually changing).
% 
% example:
% x = -1:.1:20;
% y = evaldoublegamma([1 1 1 1 2 .2 0 0],x);
% yn = y + 0.1*randn(size(y));
% [params,d,d,exitflag,output] = lsqcurvefit(@(a,b) evaldoublegamma([a 0 0],b),ones(1,6),x,yn,[],[],defaultoptimset({'OutputFcn' @outputfcnplot}));

% input
if ~exist('tol','var') || isempty(tol)
  tol = 1e-6;
end
if ~exist('numsteps','var') || isempty(numsteps)
  numsteps = 10;
end

% global stuff
global OFSC_RES;

% do it
switch state
case 'init'
  OFSC_RES = [];
case {'iter' 'done'}
  OFSC_RES = [OFSC_RES optimValues.resnorm];
  if length(OFSC_RES) >= numsteps+1
    if abs(OFSC_RES(end)-OFSC_RES(end-numsteps)) < tol
      stop = 1;
      return;
    end
  end
end

% return
stop = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % for sanity checking, we also check whether optimValues.ratio
% % is NaN; if it is, we stop the optimization.
% %
% %   if isfield(optimValues,'ratio') && isnan(optimValues.ratio) && optimValues.iteration > 5
% %     stop = 1;
% %     return;
% %   end
