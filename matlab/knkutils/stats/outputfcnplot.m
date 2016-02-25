function stop = outputfcnplot(params,optimValues,state,numiters,data,datase)

% function stop = outputfcnplot(params,optimValues,state,numiters,data,datase)
%
% <params> is the current optimization parameter
% <optimValues>,<state> are optimization state stuff
% <numiters> (optional) is to plot after every this number of iterations.
%   default: 10.
% <data> (optional) is a (row or column) vector of values indicating 
%   the data that the optimizer is fitting
% <datase> (optional) is a (row or column) vector of values, matched in 
%   size to <data>, indicating the standard errors on the data
%
% in the 'init' state, make a new figure window.
% in the 'iter' and 'done' states, update two subplots
%   every <numiters> iterations.  in the first subplot, plot
%   the parameter history.  in the second subplot,
%   plot the current parameter values.  if 'q' is pressed
%   in this figure window, stop the optimization.
%   if 'p' is pressed, issue a keyboard command.
%
% if <data> is supplied, we actually make a third subplot
% showing the data and the current model fit.  if <datase>
% is supplied, we put error bars on the data.
% 
% example:
% x = -1:.1:20;
% y = evaldoublegamma([1 1 1 1 2 .2 0 0],x);
% yn = y + 0.1*randn(size(y));
% [params,d,d,exitflag,output] = lsqcurvefit(@(a,b) evaldoublegamma([a 0 0],b),ones(1,6),x,yn,[],[],defaultoptimset({'OutputFcn' @(a,b,c)outputfcnplot(a,b,c,1,yn)}));
%
% history:
% 2011/11/26 - add <data> and <datase>
% 2010/08/25 - hack out ratio check; hack out <restol>; hack out NaN case for <numiters>

% input
if ~exist('numiters','var') || isempty(numiters)
  numiters = 10;
end
if ~exist('data','var') || isempty(data)
  data = [];
end
if ~exist('datase','var') || isempty(datase)
  datase = [];
end
data = flatten(data);
datase = flatten(datase);

% global stuff
global OUTPUTFCNPLOT_PARAMSHISTORY;

% calc
nsubplots = choose(isempty(data),2,3);

% do it
switch state
case 'init'
  OUTPUTFCNPLOT_PARAMSHISTORY = [];
  figure;
case {'iter' 'done'}
%  fprintf('%.4f ',params); fprintf('\n');
  OUTPUTFCNPLOT_PARAMSHISTORY = [OUTPUTFCNPLOT_PARAMSHISTORY; params];
  if mod(optimValues.iteration,numiters)==0
    clf;
    subplot(nsubplots,1,1); plot(OUTPUTFCNPLOT_PARAMSHISTORY); xlabel('Iteration number'); ylabel('Parameter value');
    subplot(nsubplots,1,2); bar(params); xlabel('Parameter number'); ylabel('Parameter value'); title(mat2str(params));
    if ~isempty(data)
      subplot(nsubplots,1,3); hold on;
      fit = flatten(optimValues.residual) + data;  % what is the current fit?
      r = calccod(fit,data);
      bar(data);
      if ~isempty(datase)
        errorbar2(1:length(data),data,datase,'v','r-');
      end
      plot(fit,'g-','LineWidth',2);
      ax = axis; axis([0 length(data)+1 ax(3:4)]);
      xlabel('Data point'); ylabel('Value');
      title(sprintf('R^2 (relative to mean) = %.3f',r));

% EXPERIMENTAL
%       subplot(2,2,4); hold on;
%       scatter(fit,data,'r.');
%       errorbar2(fit,data,datase,'v','r-');
%       axissquarify;

    end
    drawnow;
    if isequal(get(gcf,'CurrentCharacter'),'q')
      stop = 1;
      return;
    end
    if isequal(get(gcf,'CurrentCharacter'),'p')
      keyboard;
    end
  end
end

% return
stop = 0;
