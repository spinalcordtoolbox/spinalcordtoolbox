%demoL1iPotts_DecLap
% Reconstruction of a blurred sparse signal from incomplete measurements under
% Laplacian noise using the inverse L1-Potts functional

% create signal
groundTruth = loadSparse('sig2');
n = numel(groundTruth);

% create Gaussian convolution matrix
K = convkernel('gaussian', 51, 5);
Afull = spconvmatrix(K, n);

% select random measurements
idx = sort(randidx(numel(groundTruth), 0.5));
A = Afull(idx, :);

% create blurred and noisy signal (Laplacian noise)
fBlurry = A * groundTruth;
fNoisy = fBlurry + 0.05* randl(size(fBlurry));

% reconstruction
gamma = 0.5;
u = minL1iSpars(fNoisy, gamma, A);

% show result
showSparse(fNoisy, u, groundTruth)

