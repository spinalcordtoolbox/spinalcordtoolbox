function f = calcamplitudespectrum(im,b)

% function f = calcamplitudespectrum(im,b)
% 
% <im> is res x res x images
% <b> (optional) is the bandwidth to use in the local regression.
%   default: res/10.
%
% calculate the mean amplitude spectrum of the images
% using a Hanning window to reduce wraparound artifacts
% (the Hanning window does reduce contrast, so be wary).
% then use local regression to determine a function
% that maps spatial frequency to the mean amplitude spectrum.  
% (we regress on the log of the amplitude spectrum to make things 
% well-behaved.)  return a vector with the value of this function at 
% 0:floor(res/2*sqrt(2)).
%
% example:
% f = [];
% for p=1:100
%   im = generatepinknoise(64,1,1,1);
%   f(p,:) = calcamplitudespectrum(im);
% end
% figure; plot(0:size(f,2)-1,f,'-');
% axis([.1 100 .1 100]); axis square;
% set(gca,'XScale','log'); set(gca,'YScale','log');

% input
if ~exist('b','var') || isempty(b)
  b = size(im,1)/10;
end

% calc
res = size(im,1);
[xx,yy] = calccpfov(res);
cpfov = sqrt(xx.^2 + yy.^2);
win = hanning(res)*hanning(res)';
% win = win/sum(flatten(abs(fft2(win))))*4096;
% %win = win/mean(win(:));

% hanning window and then calculate mean amplitude spectrum
mas = mean(fftshift2(abs(fft2(bsxfun(@times,im,win)))),3);

% do local regression
f = exp(localregression(flatten(cpfov),flatten(log(mas)),0:floor(res/2*sqrt(2)),[],[],[b 1 1]));
