%demoL2iPotts_Deconv
% Reconstruction of a blurred jump-sparse signal from incomplete measurements under
% Gaussian noise using the inverse L2-Potts functional

% load signal
groundTruth = loadPcwConst('sampleDec');
n = numel(groundTruth);

% create Gaussian kernel
K = convkernel('gaussian', 51, 6);
Afull = spconvmatrix(K, numel(groundTruth));

% select random measurements
idx = sort(randidx(n, 0.5)) ;
A = Afull(idx, :);

% create blurred and noisy signal (Gaussian noise)
fBlurry = A * groundTruth(:);
fNoisy = fBlurry + 0.05 * randn(size(fBlurry));

% reconstruction using the inverse L2-Potts problem
gamma = 0.03;
[u, dataError, nJumps, energy] = minL2iPotts(fNoisy, gamma, A);

% show result
showPotts(fNoisy, u, groundTruth, 'L^2-iPotts')
