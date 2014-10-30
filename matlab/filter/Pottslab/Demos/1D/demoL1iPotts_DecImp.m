%demoL1iPotts_DecImp
% Reconstruction of a blurred jump-sparse signal from incomplete measurements under
% impulsive noise using the inverse L1-Potts functional

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
fBlurry = A * groundTruth(:);

% impulsive noise (noiseFraction = number of pixels destroyed)
noiseFraction = 0.3;
ridx = randidx(numel(fBlurry), noiseFraction);
f = fBlurry;
f(ridx) =  (rand(size(ridx)));

% Solve inverse L1-Potts problem
gamma = 0.4;
u = minL1iPotts(f, gamma, A);

% show result
showPotts(f, u, groundTruth, 'L^1-iPotts')

