%demoL2iPotts_DecLap
% Reconstruction of a blurred sparse signal from incomplete measurements under
% Laplacian noise using the inverse L2-Potts functional

% create signal
groundTruth = loadSparse('sig1');
n = numel(groundTruth);

% create Gaussian convolution matrix
K = convkernel('gaussian', 51, 5);
Afull = spconvmatrix(K, n);

% select random measurements
idx = sort(randidx(numel(groundTruth), 0.5));
A = Afull(idx, :); 

% create blurred and noisy signal (Gaussian noise)
fBlurry = A * groundTruth;
sigma = 0.05;
fNoisy = fBlurry + sigma* randn(size(fBlurry));

% reconstruction
gamma = 0.025;
[u, nSpikes] = minL2iSpars(fNoisy, gamma, A);

% show result
showSparse(fNoisy, u, groundTruth)
