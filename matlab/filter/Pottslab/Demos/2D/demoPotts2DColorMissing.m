% load image
img = double(imread('church.jpg'))/255;

% add Gaussian noise
sigma = 0.2;
imgNoisy = img + sigma * randn(size(img));

% set weights and destroy image
[m,n,l] = size(img);
missingFraction = 0.6;
weights = rand(m,n) > missingFraction;
imgNoisy(~cat(3, weights, weights, weights)) = 0; 

%% Potts restoration
gamma = 0.4;
clear opts;
opts.weights = weights; 
opts.verbose = true;
tic
u = minL2Potts2DADMM(imgNoisy, gamma, opts);
toc

%% Show result
subplot(2,2,1)
imshow(img)
title('Original')
subplot(2,2,2)
imshow(imgNoisy)
title('Noisy image and black pixels are missing')
subplot(2,2,3)
imshow(u)
title('Potts restoration')