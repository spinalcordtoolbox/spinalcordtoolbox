function r = psnr( groundTruth, signal )
%PSNR Computes the peak signal to noise ratio

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

warning('The function psnr conflicts with the new function psnr from the image processing toolbox from Matlab v2014a. Use plpsnr instead.')

mse = mean(abs(signal(:) - groundTruth(:)).^2);
r = 10 * log10(max(groundTruth(:))^2 / mse);

end

