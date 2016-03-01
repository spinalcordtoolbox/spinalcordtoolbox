function f = calchrfpeak(v,tr,window1,window2)

% function f = calchrfpeak(v,tr,window1,window2)
%
% <v> is a row or column vector with an HRF.  the first point of the HRF is
%   assumed to be time-locked to the start of the trial.
% <tr> is the TR in seconds
% <window1> (optional) is [A B] with the window within which to integrate 
%   in order to figure out the sign of the HRF.  A and B should be in seconds
%   and should be non-negative.  default: [4 9].
% <window2> (optional) is [C D] with the window within which to look for
%   the peak.  C and D should be in seconds and should be non-negative.
%   default: [0 13].
%
% return an estimate of the peak value of the HRF.  this value can
% be positive or negative, depending on whether the HRF exhibits a
% positive or negative deflection.
%
% specifically, we perform the following steps:
% 1. figure out the sign of the HRF by summing over the values within <window1>.
% 2. if the sign is negative, we sign-flip the HRF (so that it is positive).
% 3. then, considering the HRF values within <window2>, we use cubic interpolation
%    to estimate the maximum value.
% 4. finally, if the sign of the HRF was positive, we simply return the maximum 
%    value found.  if the sign of the HRF was negative, we return the negative 
%    of the maximum value found.
%
% example:
% [hrf,tr] = getsamplehrf(5);
% peak = calchrfpeak(hrf{1},tr);
% figure; hold on;
% plot(tr*(0:length(hrf{1})-1),hrf{1},'r.-');
% straightline(peak,'h','b-');

% input
if ~exist('window1','var') || isempty(window1)
  window1 = [4 9];
end
if ~exist('window2','var') || isempty(window2)
  window2 = [0 13];
end

% figure out indices to quantify the sign
signindices = 1 + (ceil(window1(1)/tr):floor(window1(2)/tr));
signindices(signindices > length(v)) = [];

% figure out the sign
thesign = signforce(sum(v(signindices)));

% figure out indices where to look for the peak
peakindices = 1 + (ceil(window2(1)/tr):floor(window2(2)/tr));
peakindices(peakindices > length(v)) = [];

% figure out the peak
thepeak = calcpeak(flatten(v(peakindices) * thesign),[],'cubic');

% return it
f = thepeak * thesign;
