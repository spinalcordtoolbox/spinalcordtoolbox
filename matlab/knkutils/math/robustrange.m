function [f,mn,mx] = robustrange(m)

% function [f,mn,mx] = robustrange(m)
%
% <m> is a matrix
%
% figure out a reasonable range for the values in <m>,
% that is, one that tries to include as many of the values
% as possible, but also tries to exclude the effects of potential
% outliers (so that we don't get a really large range).  
% see the code for specific details on how we do this.
%
% return:
%  <f> as [<mn> <mx>]
%  <mn> as the minimum value of the range
%  <mx> as the maximum value of the range
%
% example:
% x = randn(1,10000).^2;
% figure; hold on;
% hist(x,100);
% rng = robustrange(x);
% straightline(rng,'v','r-');
% title(sprintf('range is %s',num2str(rng)));

% absolute min and max
absmn = min(m(:));
absmx = max(m(:));

% percentiles
vals = prctile(m(:),[.1 10 50 90 99.9]);

% percentile-based min and max
pmn = vals(3) - 5*(vals(3)-vals(2));
pmx = vals(3) + 5*(vals(4)-vals(3));

% whether to rerun (recursively)
rerun = 0;

% deal with max
if vals(5) <= pmx  % if the 99.9 is reasonably small, use it
  if absmx <= vals(3) + 1.1*(vals(5)-vals(3))
    finalmx = absmx;  % actually, if the absolute max isn't too big, use that
  else
    finalmx = vals(5);
  end
else
  rerun = 1;  % hmm, something is funny.  probably there are outliers.  let's chop off and re-run.
  m(m>pmx) = [];
end

% deal with min
if vals(1) >= pmn
  if absmn >= vals(3) - 1.1*(vals(3)-vals(1))
    finalmn = absmn;
  else
    finalmn = vals(1);
  end
else
  rerun = 1;
  m(m<pmn) = [];
end

% rerun without outliers
if rerun
  [f,mn,mx] = robustrange(m);
  return;
end

% output
f = [finalmn finalmx];
mn = finalmn;
mx = finalmx;
