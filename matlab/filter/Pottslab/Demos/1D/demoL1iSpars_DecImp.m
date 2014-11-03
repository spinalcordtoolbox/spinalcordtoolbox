%demoL1iPotts_DecImp
% Reconstruction of a blurred sparse signal from incomplete measurements under
% impulsive noise using the inverse L1-Potts functional

% create signal
groundTruth = loadSparse('sig2');
n = numel(groundTruth);

% create Gaussian convolution matrix
K = convkernel('gaussian', 51, 5);
Afull = spconvmatrix(K, n);
% select randoom measurements
idx = sort(randidx(numel(groundTruth), 0.5));
A = Afull(idx, :); 

% create blurred and noisy signal (impulsive noise)
fBlurry = A * groundTruth;
sigma = 0.3;
ridx = randidx(numel(fBlurry), sigma);
fNoisy = fBlurry;
fNoisy(ridx) =  (rand(size(ridx)) - 0.5);

% reconstruction
gamma = 0.45;
u = minL1iSpars(fNoisy, gamma, A);

% show result
showSparse(fNoisy, u, groundTruth)


