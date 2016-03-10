function [eff,stim,X,polymatrix] = expdesignefficiency(ndistinct,nrep,polydeg,noconsecutive,design,hrf,x,polymatrix)

% function [eff,stim,X,polymatrix] = expdesignefficiency(ndistinct,nrep,polydeg,noconsecutive,design,hrf,x,polymatrix)
%
% <ndistinct> is the number of distinct trial types
% <nrep> is the number of repetitions of each trial type
% <polydeg> is the max degree of the polynomials to use
% <noconsecutive> is whether to return -Inf if two consecutive trials have the same type
% <design> is a vector of 0/1 indicating the experimental design.
%   sum(design) should be equal to <ndistinct>*<nrep>.
% <hrf> is a vector with the HRF timecourse to assume (coincident with trial onset)
% <x> is a vector with dimensions 1 x <ndistinct>*<nrep>.  we sort this vector and
%   use the indices to determine the experimental design.  the first <nrep> entries
%   in the indices get allocated to the first trial type; the second
%   <nrep> entries get allocated to the second trial type; and so on.
% <polymatrix> (optional) is a speed-up.  depends on length(<design>) and <polydeg>.
%
% return <eff> with the efficiency value.  this is determined by convolving the
%   stimulus design matrix with the <hrf>, adding the polynomial nuisance functions,
%   calculating inv(X'*X) where X is the result of the previous step, summing
%   the diagonal elements corresponding to the stimulus events, and then taking the
%   reciprocal.
% return <stim> with the stimulus design matrix, length(design) x <ndistinct>.
%   each column gives the stimulation pattern for one trial type.
% return <X> with the final design matrix, length(design) x <ndistinct>+<polydeg>+1.
%
% example:
% [eff,stim] = expdesignefficiency(2,5,2,0,repmat([0 0 1 0 0],[1 10]),[0 0 1 2 1],randn(1,10));
% figure; imagesc(stim);

% calc
time = length(design);

% deal with polymatrix
if nargin < 8 || isempty(polymatrix)
  polymatrix = constructpolynomialmatrix(time,0:polydeg);
end

% determine design
[d,ix] = sort(x);

% check for consecutive
if noconsecutive
  temp = diff(ix);
  if any(abs(temp(1:2:end))==1)
    eff = -Inf;
    stim = NaN;
    X = NaN;
    return;
  end
end

% calc indices of where to stick trials in
where = find(design);

% construct stimulus design matrix
stim = zeros(time,ndistinct);
offset = flatten(repmat(linspacefixeddiff(0,time,ndistinct),[nrep 1]));  % needed to assign different trial types to different columns
stim(where(ix) + offset) = 1;

% do the HRF convolution
X = conv2(stim,hrf');
X = X(1:time,:);

% add nuisances
X = [X polymatrix];

% calc efficiency
temp = diag(inv(X'*X));
eff = 1/sum(temp(1:ndistinct));
