function f = generatewinnermap(m,cmap)

% function f = generatewinnermap(m,cmap)
%
% <m> is A x B x N where N indicates the dimension to search over
%   and A is the dimension to normalize with respect to
% <cmap> is a colormap intended for centered colormap lookup (see cmaplookup.m)
%
% find the max value along the N dimension and convert the index of the
% max case into a color using <cmap>.  then, darken the V value of
% the color by multiplying with j/k where j is the max value itself
% and k is the maximum max value found over the A dimension.
% return RGB values in A x B x 3.
%
% example:
% figure; imagesc(generatewinnermap(rand(10,10,4),cmaphue(4)));

% search for max (A x B)
[mx,ix] = max(m,[],3);

% convert ix into HSV values (A*B x 3)
temp = rgb2hsv(squish(cmaplookup(ix,1,size(cmap,1)+1,1,cmap),2));

% normalize V with respect to max found over dimension A
temp(:,3) = temp(:,3) .* squish(bsxfun(@rdivide,mx,max(mx,[],1)),2);

% convert HSV values back to RGB (A x B x 3)
f = reshape(hsv2rgb(temp),size(m,1),size(m,2),3);
