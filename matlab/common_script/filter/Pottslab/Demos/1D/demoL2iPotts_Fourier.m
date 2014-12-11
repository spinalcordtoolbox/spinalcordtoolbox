%demoL2iPotts_Deconv
% Reconstruction of a jump-sparse signal from incomplete Fourier measurements under
% Gaussian noise using the inverse L2-Potts functional

% load signal
groundTruth = loadPcwConst('sampleDec');

% create reduced Fourier matrix 
n = numel(groundTruth);
F = fft(eye(n)) / sqrt(n);
fraction = 0.75;
idx = [1, randidx(n-1, fraction) + 1]; % random indices but keep DC component
A = F(idx, :);
fFourier = A * groundTruth;

% 
sigma = 0.05;
noise =  randn(size(groundTruth)) +  1i * randn(size(groundTruth));
fNoisy = fFourier + sigma * noise(idx);

% reconstruction with inverse L^2-Potts functional
gamma = 0.05;
u = minL2iPotts(fNoisy, gamma, A);

% show reconstruction
showPotts(fftshift(log(abs(fNoisy))), real(u), groundTruth, 'L^2-iPotts')



