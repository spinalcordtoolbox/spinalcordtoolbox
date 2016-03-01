function f = colorinterpolate(colors,n,mode)

% function f = colorinterpolate(colors,n,mode)
%
% <colors> is an M x 3 matrix of colors
% <n> is a positive integer indicating the number of entries to allocate per color.
% <mode> is
%   0 means interpolate and omit the last color
%   1 means interpolate and include the last color
%   2 means act as if first color is repeated after the last color and omit it
%   3 means act as if first color is repeated after the last color and include it
%
% make a colormap by interpolating between the colors in <colors>.
% if mode==0, return an (M-1)*<n> x 3 matrix
% if mode==1, return an (M-1)*<n>+1 x 3 matrix
% if mode==2, return an M*<n> x 3 matrix
% if mode==3, return an M*<n>+1 x 3 matrix
%
% example:
% figure; colormap(colorinterpolate([0 0 0; 1 1 1],9,1)); colorbar;

% tack on last color if necessary
if mode==2 || mode==3
  colors(end+1,:) = colors(1,:);
end

% interpolate
f = [];
for p=1:size(colors,1)-1

  % interpolate this chunk, channel by channel
  temp = [];
  for q=1:3
    temp = [temp linspacecircular(colors(p,q),colors(p+1,q),n)'];
  end

  % add it in
  f = [f; temp];

end

% include the last color if necessary
if mode==1 || mode==3
  f = [f; colors(end,:)];
end
