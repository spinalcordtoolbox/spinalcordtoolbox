function showPotts(data, rec, groundTruth, method)
%showPotts Display result of Potts reconstructions

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
    plot(groundTruth, '--r')
    hold off
    leg = [leg, {'Ground truth'}];
    title(['PSNR: ' num2str(plpsnr(groundTruth, rec)), ', #Jumps: ', num2str(countJumps(rec))] );
end

legend(leg);

end
