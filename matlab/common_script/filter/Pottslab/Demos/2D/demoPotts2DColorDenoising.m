% load image  (image source: Wikipedia, painting of Roy Lichtenstein)
img = double(imread('lookmickey.jpg'))/255;

% add Gaussian noise
sigma = 0.3;
imgNoisy = img + sigma * randn(size(img));

% Potts restoration
gamma = 0.75;
tic
u = minL2Potts2DADMM(imgNoisy, gamma, 'verbose', true);
toc

% show results
subplot(1,3,1)
imshow(img)
title('Original')
subplot(1,3,2)
imshow(imgNoisy)
title('Noisy image')
subplot(1,3,3)
imshow(u)
title('Potts restoration')