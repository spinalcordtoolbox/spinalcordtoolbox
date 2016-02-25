function [h1,h2,mn,se] = scatterline(x,y,vals,h,numboot,color,mode,wantrobust)

% function [h1,h2,mn,se] = scatterline(x,y,vals,h,numboot,color,mode,wantrobust)
%
% <x>,<y> are (row or column) vectors of the same length
% <vals> (optional) is a sorted vector of x-values to perform the regression at.
%   default: linspace(min(x),max(x),20).
% <h> (optional) is
%     X means to use local regression and to use X as the bandwidth (see localregression.m).
%   NaN means to use fitline2derror.m.
%   default: std(x)/10.
% <numboot> (optional) is the number of bootstraps to perform.  default: 100.
%   special case is 0 which means to do not perform bootstrap and just 
%   operate on entire dataset.  in this case, <h1> and <se> are returned as [].
% <color> (optional) is a 3-element vector with a color.  default: [1 0 0].
% <mode> (optional) is 0 (L2) or 1 (L1).  when <h> is X, then <mode> is used
%   in localregression.m accordingly.  when <h> is NaN, then <mode> controls
%   the exponent used in fitline2derror.m.  default: 0.
% <wantrobust> (optional) is 
%   0 means normal mode (use mean and standard deviation across bootstraps)
%   1 means robust mode (use median and percentiles across bootstraps).
%     in this mode, the standard errors that are plotted actually represent
%     15.87th and 84.13th percentiles of the bootstraps (i.e. a 68%
%     confidence interval), and <se> will be a vector of imaginary numbers,
%     where the real and imaginary parts are these percentile values,
%     respectively.
%   default: 0.
%
% draw bootstraps from <x> and <y> and perform either local linear regression 
% (when <h> is X) or linear regression (when <h> is NaN) at the values in <vals>.
% the error metric is determined by <mode>.
%
% on the current figure, draw an errorbar polygon (see errorbar3.m) that 
% indicates the mean +/- the standard deviation across bootstraps; this polygon
% gets a color that is halfway between <color> and pure white.  note that the 
% polygon may be discontinuous (and may therefore actually consist of multiple
% polygons).  also, draw a line (with LineWidth 2) that indicates the mean across
% bootstraps; this line gets the color <color>.
%
% note that we use nanmean and nanstd (or nanmedian and prctile when <wantrobust>)
% to summarize the bootstraps.  (NaNs can occur due to insufficient number of 
% points involved in the various regressions.)  however, to prevent artificially 
% small standard errors, we enforce that NaNs will be explicitly inserted 
% whenever more than half the bootstraps result in NaNs.
%
% return:
%  <h1> as the handle(s) of the errorbar polygon(s)
%  <h2> as the handle of the line
%  <mn> as a vector of means (corresponding to what is plotted)
%  <se> as a vector of standard errors (corresponding to what is plotted)
%
% example:
% x = randn(1,1000);
% y = x + randn(1,1000);
% figure;
% subplot(1,2,1); hold on;
% scatter(x,y,'k.');
% scatterline(x,y);
% subplot(1,2,2); hold on;
% scatter(x,y,'k.');
% scatterline(x,y,[],NaN);

% input
if ~exist('vals','var') || isempty(vals)
  vals = linspace(min(x),max(x),20);
end
if ~exist('h','var') || isempty(h)
  h = std(x)/10;
end
if ~exist('numboot','var') || isempty(numboot)
  numboot = 100;
end
if ~exist('color','var') || isempty(color)
  color = [1 0 0];
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if ~exist('wantrobust','var') || isempty(wantrobust)
  wantrobust = 0;
end

% hold on
prev = ishold;
hold on;

% do it
if numboot==0
  if isnan(h)
    dist = polyval(fitline2derror(flatten(x),flatten(y),[],choose(mode==0,2,1)),vals);  % 1 x points
  else
    dist = localregression(flatten(x),flatten(y),vals,[],[],h,mode);  % 1 x points
  end
  if wantrobust
    mn = nanmedian(dist,1);
  else
    mn = nanmean(dist,1);
  end
  se = [];
  h1 = [];
  h2 = plot(vals,mn,'-','LineWidth',2,'Color',color);
else
  if isnan(h)
    dist = bootstrapdim([flatten(x); flatten(y)],2,@(xx) polyval(fitline2derror(xx(1,:),xx(2,:),[],choose(mode==0,2,1)),vals),numboot,[],1);  % boot x points
  else
    dist = bootstrapdim([flatten(x); flatten(y)],2,@(xx) localregression(xx(1,:),xx(2,:),vals,[],[],h,mode),numboot,[],1);  % boot x points
  end
  bad = sum(isnan(dist),1) > size(dist,1)/2;
  if wantrobust
    mn = copymatrix(nanmedian(dist,1),bad,NaN);
    se = copymatrix([1 j]*prctile(dist,[15.87 84.13],1),bad,NaN);
    h1 = errorbar3(vals,(real(se)+imag(se))/2,range([real(se); imag(se)],1)/2,'v',(color+[1 1 1])/2);
  else
    mn = copymatrix(nanmean(dist,1),bad,NaN);
    se = copymatrix(nanstd(dist,[],1),bad,NaN);
    h1 = errorbar3(vals,mn,se,'v',(color+[1 1 1])/2);
  end
  h2 = plot(vals,mn,'-','LineWidth',2,'Color',color);
end

% hold off
if ~prev
  hold off;
end
