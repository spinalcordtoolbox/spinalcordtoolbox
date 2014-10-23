function r = plpsnr( groundTruth, signal )
%plPSNR Computes the peak signal to noise ratio

% written by M. Storath
% $Date: 2014-05-07 14:23:10 +0200 (Mi, 07 Mai 2014) $	$Revision: 90 $

mse = mean(abs(signal(:) - groundTruth(:)).^2);
r = 10 * log10(max(groundTruth(:))^2 / mse);

end