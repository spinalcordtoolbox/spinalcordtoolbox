function [im,flt,fltF] = flattenspectra(im,fltsz,wantcheck,bands)

% function [im,flt,fltF] = flattenspectra(im,fltsz,wantcheck,bands)
% 
% <im> is res x res x images
% <fltsz> is the <sz> input to fouriertospace.m
% <wantcheck> (optional) is whether to show some diagnostic figures. default: 0.
% <bands> (optional) is the bandwidths to try.
%   default: 2.^(.5:.5:log2(ceil(size(im,1)/2*sqrt(2))))
%
% basically, we calculate the mean amplitude spectrum
% of the images and apply spatial-frequency filtering in order
% to, on average, flatten the spectra of the images.
% we return the images in <im>, the filter in <flt>, and the
% Fourier-domain filter (from which <flt> is made) in <fltF>.
%
% more specifically, this is what we do:
% calculate the mean amplitude spectrum of the images
% (using a hanning window to reduce wraparound artifacts).
% then use local regression to determine a function that maps
% spatial frequency to the mean amplitude spectrum
% (we try a variety of bandwidths and choose the one that 
% minimizes cross-validation error; also, we fit to the log
% of the amplitude spectrum to make things well-behaved).
% we then construct a Fourier-domain filter that is rotationally
% symmetric (i.e. does not affect orientation) and that has a
% gain that is the reciprocal of the determined function.  
% this filter is then converted into a unit-length space-domain 
% filter. finally, we apply the filter to the images using:
%   im = imagefilter(im,flt,2);
%
% example:
% im = generatepinknoise(64,1,16,1);
% im2 = flattenspectra(im,-21,1);
% viewimages(im);
% viewimages(im2);

% input
if ~exist('wantcheck','var') || isempty(wantcheck)
  wantcheck = 0;
end
if ~exist('bands','var') || isempty(bands)
  bands = 2.^(.5:.5:log2(ceil(size(im,1)/2*sqrt(2))));
end

% calc
res = size(im,1);
[xx,yy] = calccpfov(res);
cpfov = sqrt(xx.^2 + yy.^2);
win = hanning(res)*hanning(res)';
nfold = 5;  % number of cross-validation folds
rng = 0:.5:ceil(res/2*sqrt(2));  % good SF range

% hanning window and then calculate mean amplitude spectrum (note that windowing mangles the contrast)
mas = mean(fftshift2(abs(fft2(bsxfun(@times,im,win)))),3);

% do local regression
if length(bands) > 1
  [hbest,err] = localregressionbandwidth(flatten(cpfov),flatten(log(mas)),bands,rng,nfold);
else
  hbest = bands;
end

% construct filter
fltF = 1 ./ ifftshift2(reshape(exp(localregression(flatten(cpfov),flatten(log(mas)),flatten(cpfov),[],[],hbest)),size(cpfov)));
flt = fouriertospace(fltF,fltsz,wantcheck);  % note unit-length filter is obtained, so the contrast change above doesn't matter

% do quick check
if ~all(isfinite(fltF(:)))
  fprintf('warning! not all of the fourier-domain filter is finite. did the local regression return NaNs?\n');
end

% apply filter
im = imagefilter(im,flt,2);

% inspect results
if wantcheck

  if exist('err','var')
    drawnow; figure; plot(err,'bo-');
    xlabel('bandwidth'); ylabel('error');
    title('determination of optimal bandwidth in local regression');
  end

  drawnow; figure; hold on;
  scatter(flatten(cpfov),flatten(log(mas)),'r.');
  plot(rng,localregression(flatten(cpfov),flatten(log(mas)),rng,[],[],hbest),'k-','LineWidth',3);
  xlabel('cycles per FOV'); ylabel('log of mean amplitude spectrum');
  title('local regression fit using optimal bandwidth');

end
