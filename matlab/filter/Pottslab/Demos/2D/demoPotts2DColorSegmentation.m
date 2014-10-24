% load image  (image source: The Berkeley Segmentation Dataset and Benchmark)
img = double(imread('desert.jpg'))/255;
scale = 0.5;
img = imresize(img, scale);

% Potts restoration
gamma = 2 * scale;
tic;
u = minL2Potts2DADMM(img, gamma, 'verbose', true);
toc

%%
subplot(1,2,1)
imshow(img)
energyImg = energyL2Potts(img, img, gamma);
title(sprintf('Original (Potts energy: %.1f)', energyImg));

%%
subplot(1,2,2)
imshow(u)
energyU = energyL2Potts(u, img, gamma);
title(sprintf('Potts segmentation (Potts energy: %.1f)', energyU));