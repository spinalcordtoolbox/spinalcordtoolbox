function f = cmapdistinct(n)

% function f = cmapdistinct(n)
%
% <n> is desired number of entries.  must be a positive integer between 1 and 15.
%
% return a hue-based colormap.  we manually selected 9 distinct hues (maximum saturation
% and lightness).  we will return these hues up to the number of colors requested.
% if you requested more than 9 colors, then we will return a second set of 6
% hues with the lightness reduced.  if you request more than 15 colors, we die.
%
% example:
% figure; imagesc((1:15)'); axis equal tight; colormap(cmapdistinct(15));

% constants
          %red       %orange     %yellow     %lightgreen  %cyan        %lightblue    %darkblue   %purple      %magenta
colors =  [0/360 1 1; 30/360 1 1; 60/360 1 .95;   120/360 1 1;   180/360 1 1; 205/360 1 1; 230/360 1 1;   275/360 1 1; 300/360 1 1];  % HSV
colors2 = [0/360 1 .45;           60/360 1 .45; 120/360 1 .45; 180/360 1 .45;            230/360 1 .45;              300/360 1 .45];  % HSV

% calc
nn = size(colors,1);

% sanity check
assert(n >= 1 && n <= 15);

% do it
if n <= nn
  f = [colors(1:n,:)];
else
  f = [colors; colors2(1:(n-nn),1) colors2(1:(n-nn),2) colors2(1:(n-nn),3)];
end
f = hsv2rgb(f);
