function [f,dist] = calcnoiseceiling(mn,se,n1,n2,metric)

% function [f,dist] = calcnoiseceiling(mn,se,n1,n2,metric)
%
% <mn> and <se> are the same dimensions, cases x amplitudes.  <mn> should be the
%   estimated mean and <se> should be the estimated standard error.  each case 
%   is treated independently.  what we are trying to do is to estimate how well
%   we could ever expect to predict <mn> given the level of noise in the data.
% <n1> (optional) is the number of signal simulations to perform.  default: 1000.
% <n2> (optional) is the number of measurement simulations to perform 
%   for each signal simulation.  default: 10.
% <metric> (optional) is a function that accepts two matrices of the same dimensions
%   and calculates a metric along the second dimension.  default is @(x,y) calccod(x,y,2,0,0)
%   which means to calculate R^2 without mean-subtraction.
%
% calculate the values of <metric> that we would expect to obtain between a signal and 
% noisy measurements of this signal.  the characteristics of the signal and noisy measurements
% are matched to what is observed in <mn> and <se>.  specifically, we assume
% that the signal is normally distributed with a certain mean and standard deviation, 
% that the noise is normally distributed with zero mean and some other standard deviation, 
% and that the noisy measurements of the signal are the summation of the signal and the noise.  
% we attempt to estimate the various means and standard deviations from <mn> and <se>.
%
% the process is accomplished in the following way:
%   1. we presume that the noise is normally distributed and in the same way for all of the
%      different amplitudes.  so, we calculate the pooled standard error (square root of the
%      sum of the squares of the standard errors in each case), giving us the
%      standard error of the measurement process.
%   2. we calculate the mean of the amplitudes in each case.  this yields an estimate of the true
%      signal mean.
%   3. we calculate the total variance of the amplitudes in each case.  we subtract the estimated 
%      variance due to noise in the measurement process (which is simply the square of the 
%      value obtained in step 1).  after subtraction, any negative values are rectified to 0.
%      this yields an estimate of the true signal variance.  taking the square root yields an 
%      estimate of the true signal standard deviation.
%   4. finally, we perform Monte Carlo simulations in which:
%     4a. we draw random samples from a normal distribution whose mean is matched to the
%         true signal mean and whose standard deviation is matched to the true signal standard 
%         deviation.  this represents the signal.
%     4b. we add normally distributed noise to the signal.  the noise comes from a 
%         normal distribution with zero mean and with standard deviation matched to 
%         the standard error of the measurement process (calculated in step 1).
%     4c. we calculate the <metric> between the signal and the measurement.
%     4d. we repeat 4b-4c a total of <n2> times.  then we repeat 4a-4c a
%         total of <n1> times.
%
% return:
%  <f> as cases x 1 with the median of the obtained <metric> values
%  <dist> as cases x n1*n2 with the obtained <metric> values
%
% history:
% - 2013/05/12 - add parfor
% - 2011/03/06 - total revamp.
%
% example:
% data = 1+randn(10,31);
% datamn = mean(data,1);
% datase = std(data,[],1)/sqrt(10);
% [f,dist] = calcnoiseceiling(datamn,datase);
% figure; hold on;
% bar(1:31,datamn);
% errorbar2(1:31,datamn,datase,'v','r-');
% title(sprintf('noise ceiling is %.2f',f));
% figure; hold on;
% hist(dist);
% straightline(f,'v','r-');

% input
if ~exist('n1','var') || isempty(n1)
  n1 = 1000;
end
if ~exist('n2','var') || isempty(n2)
  n2 = 10;
end
if ~exist('metric','var') || isempty(metric)
  metric = @(x,y) calccod(x,y,2,0,0);
end

% calc
ncases = size(mn,1);
namp = size(mn,2);

% calc pooled noise variance (due to the measurement process)
noisevar = mean(se.^2,2);  % cases x 1

% calc the overall mean
overallmn = mean(mn,2);  % cases x 1

% calc the overall variance observed
overallvar = var(mn,[],2);  % cases x 1

% calc the true signal std (we rectify because we can't have negative std devs)
signalstd = sqrt(posrect(overallvar-noisevar));  % cases x 1

% perform Monte Carlo simulations
dist = zeros(ncases,n1,n2);
fprintf('calculating noise ceiling');
parfor x=1:n1
  statusdots(x,n1);

  % randomly generate some "true" signals (cases x amps).
  % we do this by randomly sampling from a distribution with mean overallmn and std dev signalstd.
  signal = bsxfun(@plus,bsxfun(@times,randn(ncases,namp),signalstd),overallmn);

  % then generate some noise.  it is zero-mean and has std dev set to sqrt(noisevar).  (cases x amps x n2)
  noise = bsxfun(@times,randn(ncases,namp,n2),sqrt(noisevar));

  % then simulate the measurement of the signal (cases x amps x n2).
  measurement = bsxfun(@plus,signal,noise);

  % then calculate the metric between the true signal and the measurement (cases x 1 x n2)
  dist(:,x,:) = feval(metric,repmat(signal,[1 1 n2]),measurement);

end
fprintf('done.\n');

% return
dist = reshape(dist,ncases,n1*n2);
f = median(dist,2);
