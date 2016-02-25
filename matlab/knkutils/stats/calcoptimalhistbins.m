function n = calcoptimalhistbins(x,numfolds,maxbins)

% function n = calcoptimalhistbins(x,numfolds,maxbins)
%
% <x> is a matrix of values
% <numfolds> (optional) is number of folds to perform.  default: 20.
% <maxbins> (optional) is the maximum number of bins to try.
%   default is [] which means to try up to N bins, where N is
%   range(x(:))/(iqr(x(:))/1000).
%
% evaluate different numbers of bins to use when performing hist.m on <x>.
% to evaluate goodness, we use n-fold cross-validation.  on each fold,
% we interpret the result of applying hist to the training data as
% a probability distribution and calculate the log-likelihood of the
% testing data.  after completing the folds, we calculate the average
% log-likelihood of the data points.  we return the number of bins
% that achieves the maximum average log-likelihood.
%
% some details:
% - the minimum number of bins we test is 2, and we increment from 2 up to
%   <maxbins> in 0.25 log2-units.
% - we use bin centers that are linearly spaced between the minimum value
%   in <x> and the maximum value in <x>.
% - for data points that fall into zero zones (i.e. a histogram bin
%   that was empty for the training data), we assume a log-likelihood
%   of -30.  this is a bit arbitrary.
% 
% example:
% nums = [100 500 1000 5000];
% for p=1:length(nums)
%   x = randn(1,nums(p));
%   drawnow; figure; hist(x,calcoptimalhistbins(x));
% end

% internal constants
badval = -30;  % what log value to use for out of range values or zero histogram bins?

% input
if ~exist('numfolds','var') || isempty(numfolds)
  numfolds = 20;
end
if ~exist('maxbins','var') || isempty(maxbins)
  maxbins = range(x(:))/(iqr(x(:))/1000);
end

% calc
numbins = round(2.^(log2(2):0.25:log2(maxbins)));  % bins to test
mn = min(x(:));
mx = max(x(:));

% do it
rec = zeros(1,length(numbins));
for p=1:length(numbins)

  % calculate log-likelihood of each data point
  probs = zeros(size(x));
  for q=1:numfolds
  
    % split the data
    [f,idx,fnot] = picksubset(x,[numfolds q]);  % deterministic
    
    % hist of the training data
    [nums,centers] = hist(fnot,linspace(mn,mx,numbins(p))); assert(length(centers) >= 2);
    spacing = centers(2) - centers(1);
    
    % calculate the probability distribution
    probdist = nums / sum(nums) / spacing;
    
    % for the testing data points, find the closest bins
    [d,ii] = min(abs(bsxfun(@minus,centers',f)),[],1);
    
    % calculate the log probability of the testing data
    tt = probdist(ii);
    results = log(tt);
    results(tt==0) = badval;  % if the histogram was 0, then need to assign a finite small value
    
    % final value
    probs(idx) = results;
    
  end
  
  % record the final probability (mean of the logs)
  rec(p) = mean(probs(:));

end

% what is the optimum?
[d,ii] = max(rec);
n = numbins(ii);

% sanity check
if ii==1 || ii==length(numbins)
  warning('we may not have found an optimum!');
end







% % DEBUGGING:
% drawnow; figure; plot(rec,'ro-');

%set(gca,'XTick',1:length(numbins),'XTickLabel',mat2cellstr(numbins));

   %ok = issorted(rec(1:ii)) & issorted(fliplr(rec(ii:end))) & ii~=1 & ii~=length(numbins);


%     % need to specially handle data points that are out of the range of the histogram
%     outofrange = f < centers(1)-spacing/2 | f > centers(end)+spacing/2;
      %  | outofrange
