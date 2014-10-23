%demoL1iPotts_DecLap
% Reconstruction of a blurred jump-sparse signal from incomplete measurements under
% Laplacian noise by the inverse L1-Potts functional

% load signal
groundTruth = loadPcwConst('sampleDec');
n = numel(groundTruth);

% create Gaussian kernel
K = convkernel('gaussian', 51, 6);

% create measurement matrix
Afull = spconvmatrix(K, numel(groundTruth));
fraction = 0.5;
idx = sort(randidx(n, fraction)) ;
A = Afull(idx, :);

% create blurred signal
fBlurry = A * groundTruth;

% add Laplacian noise of std. dev. sigma
sigma = 0.05;
fNoisy = fBlurry + sigma * randl(size(fBlurry));

% Solve inverse L1-Potts problems
gamma = 0.7;
u = minL1iPotts(fNoisy, gamma, A);

% show result
showPotts(fNoisy, u, groundTruth, 'L^1-iPotts')


