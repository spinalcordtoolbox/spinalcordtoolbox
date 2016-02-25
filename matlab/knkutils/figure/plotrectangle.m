function h = plotrectangle(bounds,style)

% function h = plotrectangle(bounds,style)
%
% <bounds> is [rmin rmax cmin cmax]
% <style> is like 'g-'
%
% return the handle to a line object.
%
% example:
% figure; plotrectangle([1 2 10 11],'m-'); set(gca,'YDir','reverse');

h = plot(bounds([3 3 4 4 3]),bounds([1 2 2 1 1]),style);
