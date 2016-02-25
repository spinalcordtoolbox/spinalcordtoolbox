function [params,R2] = fitline2derror(x,y,params0,expt)

% function [params,R2] = fitline2derror(x,y,params0,expt)
%
% <x>,<y> are row vectors of the same size
% <params0> (optional) is an initial seed [A B] (see below).
%   default is [] which means use the OLS solution.
% <expt> (optional) is the exponent for the error term.  default: 2.
%
% using lsqnonlin.m, fit a line to the points given by <x> and <y>.
% the minimized error is the sum of the perpendicular distances 
% to the line from the points, where each distance is raised to the 
% <expt> power.  for example, when <expt> is 2, we are minimizing
% the squared error, where error is calculated with respect to
% both dimensions (not just <y> as in the usual case).
%
% beware of the case where the values in <x> are all the same.
% in this case, the optimal line is a perfectly vertical line,
% and the returned parameters may take on wacky values.
%
% return:
%  <params> is [A B] where the model is A*x+B
%  <R2> is the R^2 between fitted and actual values (see calccod.m).
%    note that this value does not exactly match the point of this
%    function since it assumes the variance to explain is on
%    the y-coordinate!
%
% example:
% x = sort(randn(1,100));
% xrng = [min(x) max(x)];
% y = 5*x + 2;
% y = y + 2*abs(randn(1,length(y))).^3;
% params = polyfit(x,y,1);
% paramsB = fitl1line([x' ones(length(x),1)],y');
% paramsC = fitline2derror(x,y,[],2);
% paramsD = fitline2derror(x,y,[],1);
% mf = polyval(params,xrng);
% mfB = polyval(paramsB,xrng);
% mfC = polyval(paramsC,xrng);
% mfD = polyval(paramsD,xrng);
% figure; hold on;
% scatter(x,y,'k.');
% h = plot(xrng,mf,'r-');
% hB = plot(xrng,mfB,'g-');
% hC = plot(xrng,mfC,'b-');
% hD = plot(xrng,mfD,'c-');
% legend([h hB hC hD],{'polyfit' 'l1line' '2derror expt 2' '2derror expt 1'});

% input
if ~exist('params0','var') || isempty(params0)
  params0 = [];
end
if ~exist('expt','var') || isempty(expt)
  expt = 2;
end

% define options
options = optimset('Display','off','FunValCheck','on','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-6,'TolX',1e-6);

% define seed
if isempty(params0)
  params0 = polyfit(x,y,1);
end

% do it
  % we need the sqrt because lsqnonlin squares the error terms for us!
[params,d,d,exitflag,output] = lsqnonlin(@(a) sqrt(calcdistpointline(a,x,y).^expt),params0,[],[],options);
assert(exitflag > 0);

% how well did we do?
R2 = calccod(polyval(params,x),y);
