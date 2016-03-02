function [f,n] = hist1dimage(x,numbins,xloc,xwidth,wantmdse)

% function [f,n] = hist1dimage(x,numbins,xloc,xwidth,wantmdse)
%
% <x> is a matrix of values
% <numbins> (optional) is
%   (1) the number of histogram bins to use
%   (2) a vector of bin centers to use.
%   (3) -1 means to use calcoptimalhistbins.m to 
%       figure out the number of bins
%   default: 10.
% <xloc> (optional) is the x-location around which to 
%   center the 1D image.  default: 1.
% <xwidth> (optional) is the total width for the 1D image.
%   default: 0.5.
% <wantmdse> (optional) is
%   0 means do nothing special
%   1 means to plot an additional marker ('w.') with error bars 
%     ('w-') indicating the median +/- 1 SE (calculated using 
%     calcmdse.m)
%   2 means to plot an additional marker ('ro') with error bars
%     (r-') indicating the mean +/- 1 SD.
%   default: 1.
%
% draw a vertical 1D histogram on the current figure, 
% returning a handle to an image object in <f> and the 
% vector of counts in <n>.  what we basically do is 
% to use hist.m but plot the results as an image.
% the image that we plot is normalized to the range [0,1],
% and we use a up-down flipped gray(256) colormap.  thus,
% black indicates the maximum density, while white indicates
% no density.
%
% IMPORTANT NOTE: we have encounted issues with .eps files
% created from figures that use hist1dimage.m and that use
% cropping.  basically, the 1D images and their masks get
% out-of-whack.  be careful.
%
% example:
% x = randn(1,10000);
% figure; hist(x,20);
% figure; hist1dimage(x,20);

% input
if ~exist('numbins','var') || isempty(numbins)
  numbins = 10;
end
if ~exist('xloc','var') || isempty(xloc)
  xloc = 1;
end
if ~exist('xwidth','var') || isempty(xwidth)
  xwidth = 0.5;
end
if ~exist('wantmdse','var') || isempty(wantmdse)
  wantmdse = 1;
end
if isequal(numbins,-1)
  numbins = calcoptimalhistbins(x);
end

% hold on
prev = ishold;
hold on;

% do it
[n,centers] = hist(x(:),numbins);
f = imagesc(n(:) / max(n),[0 1]);
caxis([0 1]);
colormap(flipud(gray(256)));
set(f,'XData',xloc + xwidth/6*[-1 1]);
set(f,'YData',centers);
set(gca,'YDir','normal');

% do the marker
switch wantmdse
case 1
  result = calcmdse(x,0);
  scatter(xloc,real(result),'w.');
  errorbar2(xloc,real(result),imag(result),'v','w-');
case 2
  scatter(xloc,mean(x),'ro');
  errorbar2(xloc,mean(x),std(x),'v','r-');
end

% hold off
if ~prev
  hold off;
end



% %   nb = 2;
% %   while 1
% %     [n,centers] = hist(x(:),linspace(numbins{2}(1),numbins{2}(2),round(nb)));
% %     if mean(n(n~=0)) <= numbins{1}
% %       break;
% %     end
% %     nb = nb * 1.1;
% %   end
% 
%   nb = 2.^(1:20);
%   for p=1:length(nb)
%     for q=1:10
%       [f,ix] = picksubset(x,[10 q]);
%       x(setdiff(1:numel(x),ix)
%     [n,centers] = hist(x(:),linspace(numbins{2}(1),numbins{2}(2),round(nb)));
%     if mean(n(n~=0)) <= numbins{1}
%       break;
%     end
%     nb = nb * 1.1;
%   end
% 
