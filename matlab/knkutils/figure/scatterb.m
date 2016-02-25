function [h1,h2] = scatterb(x,y,edges,estring,ptstring,bootsz,minnum,mode)

% function [h1,h2] = scatterb(x,y,edges,estring,ptstring,bootsz,minnum,mode)
%
% <x> is a matrix with x-values.
% <y> is a matrix with y-values.  should be the same size as <x>.
% <edges> is a vector with finite, monotonically increasing values.
%   all v that satisfy edges(i) <= v < edges(i+1) belong to the ith
%   bin, whose position is taken to be the average of edges(i) and edges(i+1).
% <estring> (optional) is a plot string for the main line and error bar lines.  default: 'r-'.
% <ptstring> (optional) is a scatter string for the percentile markers.  default: 'ro'.
%   if 0, then omit the percentile markers.  in this case, <h2> is returned as [].
% <bootsz> (optional) is the <sz> argument to bootstrap.m
% <minnum> (optional) is the minimum number of data points that must be
%   in a bin in order for us to plot that bin.  default: 10.
% <mode> (optional) is
%   0 means normal (calculate standard errors as the standard deviation across bootstraps)
%   1 means calculate standard errors as the 68% confidence interval on bootstraps.
%     this is appropriate in cases where the bootstrap distribution 
%     is non-Gaussian (e.g. asymmetric).
%   default: 0.
%
% in existing figure window (if any), set hold on and draw a binned scatter plot.
% first, we plot a main line that joins the median of each bin.
% second, we plot for each bin the median and the standard error on this median as error bars.
% third, we draw markers indicating the 25th and 75th percentile of each bin.
% return <h1> as a vector of handles to the main lines and error bar lines.
% return <h2> as a vector of handles to the markers.
%
% we deal with NaNs gracefully.
%
% example:
% figure; scatterb(randn(1,1000),randn(1,1000),-3:.5:3);

% input
if ~exist('estring','var') || isempty(estring)
  estring = 'r-';
end
if ~exist('ptstring','var') || isempty(ptstring)
  ptstring = 'ro';
end
if ~exist('bootsz','var') || isempty(bootsz)
  bootsz = [];
end
if ~exist('minnum','var') || isempty(minnum)
  minnum = 10;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% calculate location and median and standard error
loc = []; md = []; se = []; p25 = []; p75 = [];
for p=1:length(edges)-1

  % figure out indices of data points in this bin
  ok = find(x >= edges(p) & x < edges(p+1));

  % location of this bin
  loc = [loc (edges(p)+edges(p+1))/2];

  % deal with case of too few values
  if length(ok) < minnum
    md = [md NaN];
    se = [se NaN];
    p25 = [p25 NaN];
    p75 = [p75 NaN];
    continue;
  end
  
  % calc
  md = [md nanmedian(y(ok))];
  temp = bootstrap(y(ok),@nanmedian,[],bootsz);
  if mode==0
    se = [se nanstd(temp)];
  else
    se = [se prctile(temp,[15.87 84.13])*[1;j]];
  end
  p25 = [p25 prctile(y(ok),25)];
  p75 = [p75 prctile(y(ok),75)];

end

% plot things
hold on;
h1 = plot(loc,md,estring);
if mode==0
  h1 = [h1 errorbar2(loc,md,se,'v',estring)];
else
  h1 = [h1 errorbar2(loc,(real(se)+imag(se))/2,range([real(se); imag(se)],1)/2,'v',estring)];
end
if isequal(ptstring,0)
  h2 = [];
else
  h2 = scatter(loc,p25,ptstring);
  h2 = [h2 scatter(loc,p75,ptstring)];
end
