function showSparse(data, rec, groundTruth, method)
%showSparse Display result of sparse reconstructions

% written by M. Storath
% $Date: 2012/10/29 01:19:08 $	$Revision: 0.1 $

% data
subplot(1,2,1)
plot(data)
title('Data')

% reconstruction
subplot(1,2,2)
plot(rec, '.', 'MarkerSize', 10)

% legend
if ~exist('method', 'var')
    leg = {'Reconstruction'};
else
    leg = {method};
end

% add groundTruth
if exist('groundTruth', 'var')
    hold on
    stem(groundTruth, 'r')
    hold off
    leg = [leg, {'Ground truth'}];
    title(['PSNR: ' num2str(plpsnr(groundTruth, rec))] );
end

legend(leg);

end
