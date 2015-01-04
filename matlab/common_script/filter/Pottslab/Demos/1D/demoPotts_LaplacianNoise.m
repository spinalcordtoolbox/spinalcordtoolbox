%% Potts-denoising, Laplacian noise

%%
% Load signal
original = loadPcwConst('sample1');
% Add Laplacian noise with parameter $\lambda = 0.1$
f = original + 0.1 * randl(size(original));

%%
% $L^2$ Potts estimator
pottsL2 = minL2Potts(f, 0.6);
figure(1)
showPotts(f, pottsL2, original,  'L2-Potts')

%%
% $L^1$ Potts estimator
pottsL1 = minL1Potts(f, 0.8);
figure(2)
showPotts(f, pottsL1, original, 'L1-Potts')