function images = performfreqwhitening(images)

% function images = performfreqwhitening(images)
%
% <images> is a 3D matrix with different images along the third dimension
%
% divide images by the mean amplitude spectrum so that the 
% mean amplitude spectrum of the result is flat (all ones).
% 
% a special case is when the DC of the mean amplitude spectrum is 
% essentially 0 (that is, absolute value less than 1e-5).
% in this case, we do not touch the DC component (and thus, it
% remains essentially 0).
%
% NOTE: this procedure assumes circular images, so there is the potential
% for bad wrap-around/edge effects when images aren't necessarily circular.
% to deal with these effects, we would need to apply a window (e.g. Hanning) 
% and filter in the space-domain and deal with all the tricky issues that 
% arise therein.
%
% example:
% images = generatepinknoise(32,[],1000,1);
% a = fftshift(mean(abs(fft2(images)),3));
% figure; imagesc(a,[0 30]); axis equal tight; colorbar;
% images2 = performfreqwhitening(images);
% b = fftshift(mean(abs(fft2(images2)),3));
% figure; imagesc(b,[0 1]); axis equal tight; colorbar;
% figure; subplot(1,2,1); imagesc(images(:,:,1)); axis equal tight;
%         subplot(1,2,2); imagesc(images2(:,:,1)); axis equal tight;

% do it
imagesF = fft2(images);
meanamp = mean(abs(imagesF),3);
if allzero(meanamp(1,1))  % if DC is zero, do not touch it
  meanamp(1,1) = 1;
end
images = real(ifft2(bsxfun(@rdivide,imagesF,meanamp)));
