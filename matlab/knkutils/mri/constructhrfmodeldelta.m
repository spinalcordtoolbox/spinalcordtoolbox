function f = constructhrfmodeldelta(duration,tr)

% function f = constructhrfmodeldelta(duration,tr)
%
% <duration> is the desired HRF duration in seconds
% <tr> is the TR
%
% return a matrix of delta basis functions (time x basis).
% the first time point is assumed to be coincident with the
% trial onset and is constrained to be always zero.  the 
% remaining time points are modeled using delta basis functions.
% we include enough basis functions to model the
% HRF up to <duration> seconds after the trial onset.
%
% example:
% figure; imagesc(constructhrfmodeldelta(21,2));

numbasis = floor(duration/tr);
f = [zeros(1,numbasis); eye(numbasis)];
