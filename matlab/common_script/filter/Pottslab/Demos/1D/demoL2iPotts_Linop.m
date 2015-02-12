%demoL2iPotts_Deconv
% Reconstruction of a blurred jump-sparse signal from incomplete measurements under
% Gaussian noise using the inverse L2-Potts functional
%
% Here it is demonstrated how to handle it if the operator A is given by a
% function

% load signal
groundTruth = loadPcwConst('equidistant');
n = numel(groundTruth);

% create Gaussian kernel
K = convkernel('gaussian', 51, 6);
Afull = spconvmatrix(K, numel(groundTruth));

% select random measurements
idx = sort(randidx(n, 0.5)) ;
A = Afull(idx, :);

% create linear operator
Aeval = @(x) A * x; %(evaluation)
Atrans = @(x) A' * x; %(evaluation of conj. transpose of A)
B = linop( Aeval, Atrans );

% create blurred and noisy signal (Gaussian noise)
fBlurry = B * groundTruth(:);
fNoisy = fBlurry + 0.05 * randn(size(fBlurry));

% reconstruction using the inverse L2-Potts problem
gamma = 0.03;
[u, dataError, nJumps, energy] = minL2iPotts(fNoisy, gamma, B);

% show result
showPotts(fNoisy, u, groundTruth, 'L^2-iPotts')
