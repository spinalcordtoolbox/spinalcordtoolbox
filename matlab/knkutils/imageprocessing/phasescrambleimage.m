function f = phasescrambleimage(f,cohs)

% function f = phasescrambleimage(f,cohs)
%
% <f> is res x res x images
% <cohs> is a vector of coherence levels in [0,100].
%   100 means completely the original image.
%   0 means completely the random-phase image.
%
% blend each image with a version of itself that has random phase
% (excluding the DC component, which we do not touch).
% each image has its own random-phase image.
% the same random-phase image is used for different blends.
% the result has dimensions res x res x images*length(cohs).
% beware of wraparound edge effects!
%
% example:
% figure; imagesc(phasescrambleimage(getsampleimage,50)); axis equal tight;

% calc
res = size(f,1);
numim = size(f,3);
temp = fft2(f);
mag = abs(temp);   % magnitude
ph = angle(temp);  % phase

% more
ph2 = angle(generaterandomphase(res,numim));  % random phase
ph2(1,1,:) = 0;                               % ensure mean stays constant
phdiff = circulardiff(ph2,ph,2*pi);           % angle needed to reach random phase

% do it  
f = zeros(res,res,numim*length(cohs));
for p=1:length(cohs)
  f(:,:,(p-1)*numim+(1:numim)) = real(ifft2(mag .* exp(j * (ph + (1-cohs(p)/100)*phdiff))));
end
