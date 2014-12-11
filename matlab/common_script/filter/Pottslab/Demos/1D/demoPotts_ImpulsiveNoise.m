%% Potts-denoising, impulsive noise

%%
% Load signal
original = loadPcwConst('sample1');
% Impulsive noise, 30 % corrupted 
f = original;
idx = randidx(size(f), 0.3);
f(idx) = rand(size(idx));

%%
% $L^2$ Potts estimator
figure(1)
pottsL2 = minL2Potts(f, 0.3);
showPotts(f, pottsL2, original, 'L2-Potts')

%%
% $L^1$ Potts estimator
figure(2)
pottsL1 = minL1Potts(f, 1);
showPotts(f, pottsL1, original, 'L1-Potts')
