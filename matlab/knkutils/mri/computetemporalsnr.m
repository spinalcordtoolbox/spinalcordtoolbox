function f = computetemporalsnr(vols)

% function f = computetemporalsnr(vols)
%
% <vols> is X x Y x Z x T with a 3D volume over time.
%
% return a matrix of size X x Y x Z with the temporal SNR.
% the temporal SNR is defined as follows:
% 1. first regress out a constant and a line from the time-series
%    of each voxel.
% 2. then compute the absolute value of the difference between each
%    pair of successive time points (if there are N time points,
%    there will be N-1 differences).
% 3. compute the median absolute difference.
% 4. divide by the mean of the original time-series and multiply by 100.
% 5. if any voxel had a negative mean, just return the temporal SNR as NaN.
%
% the purpose of the differencing of successive time points is to be relatively
% insensitive to actual activations (which tend to be slow), if they exist.
%
% if <vols> is [], we return [].
%
% example:
% vols = getsamplebrain(4);
% figure; imagesc(makeimagestack(5-computetemporalsnr(vols))); caxis([0 5]); colormap(jet); colorbar;

% internal constants
maxtsnrpolydeg = 1;

% do it
if isempty(vols)
  f = [];
else
  f = negreplace(median(abs(diff( ...
        reshape((projectionmatrix(constructpolynomialmatrix(size(vols,4),0:min(size(vols,4)-1,maxtsnrpolydeg)))*squish(vols,3)')',size(vols)), ...
        1,4)),4)./mean(vols,4) * 100,NaN);
end
