function f = constructhrfmodeldct(duration,tr,cutoff)

% function f = constructhrfmodeldct(duration,tr,cutoff)
%
% <duration> is the desired HRF duration in seconds
% <tr> is the TR
% <cutoff> (optional) is the maximum number of cycles per second.
%   can be Inf which means to include all basis functions.
%   default: 0.2.
%
% return a matrix of DCT-II basis functions (time x basis).
% the first time point is assumed to be coincident with the
% trial onset and is constrained to be always zero.  the 
% remaining time points (up to <duration> seconds after
% the trial onset) are modeled using DCT-II basis functions.
% we include only DCT-II basis functions that have a 
% frequency less than or equal to <cutoff>.
%
% note that the basis functions are not normalized.
%
% example:
% figure; imagesc(constructhrfmodeldct(50,1.2));

% input
if ~exist('cutoff','var') || isempty(cutoff)
  cutoff = 0.2;
end

% do it
numtime = floor(duration/tr);
maxk = min(floor(2*(cutoff*(numtime*tr))),numtime-1);
f = [zeros(1,maxk+1); constructdctmatrix(numtime,0:maxk)];
