%% Potts-denoising, Gaussian noise

%%
% Load signal
original = loadPcwConst('sample1');
% Add Gaussian noise with parameter sigma = 0.1
f = original + 0.1 * randn(size(original));

%%
% Compute $L^2$ Potts estimator
figure(1)
pottsL2 = minL2Potts(f, 0.1);
showPotts(f, pottsL2, original, 'L2-Potts')

%%
% Compute $L^1$ Potts estimator
figure(2)
pottsL1 = minL1Potts(f, 0.4);
showPotts(f, pottsL1, original, 'L1-Potts')
