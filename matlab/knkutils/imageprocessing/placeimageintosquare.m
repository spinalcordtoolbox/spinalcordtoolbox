function f = placeimageintosquare(im,bg,res,r,c,sz)

% function f = placeimageintosquare(im,bg,res,r,c,sz)
%
% <im> is an N x N image
% <bg> is a value to use for the background
% <res> is the number of pixels along one side of the overall image
% <r> is the row associated with the center of the image (can be a decimal).
%   if [], default to the exact center of the overall image along the vertical dimension.
% <c> is the column associated with the center of the image (can be a decimal).
%   if [], default to the exact center of the overall image along the horizontal dimension.
% <sz> is the desired length of a side of the image in pixels (can be a decimal).
%
% place <im> into the portion of the overall image specified by <r>, <c>, and <sz>.
% we take great pains to ensure that the field-of-view is exactly preserved.
%
% some details on the implementation: we first interpolate through the original
% image, upsampling slightly, using cubic interpolation.  then we downsample
% the result using lanczos3 interpolation in order to achieve exactly the desired
% resolution.  we use some padding to ensure that the interpolation results are
% sane (finite).  also, we impose a hard crop at the very last step to ensure that
% the interpolated image does not extend far beyond the ideal boundaries.  the crop
% we use is simply like round(r-sz/2); this has the consequence that the boundaries
% of the image are soft for at most one pixel (and can be zero if a given boundary
% exactly coincides with the edge of a pixel in the overall image).
%
% example:
% a = getsampleimage;
% figure;
% subplot(2,2,1); imagesc(placeimageintosquare(a,0.5,128,60,40,20),[0 1]); axis equal tight;
% subplot(2,2,2); imagesc(placeimageintosquare(a,0.5,128,100,40,20),[0 1]); axis equal tight;
% subplot(2,2,3); imagesc(placeimageintosquare(a,0.5,128,60,40,40),[0 1]); axis equal tight;
% subplot(2,2,4); imagesc(placeimageintosquare(a,0.5,128,60,40,100),[0 1]); axis equal tight;

% calc
n = size(im,1);

% we will use this many extra pixels on all sides to make sure that the interpolation will produce valid values.
% at least 5 pixels so the first interpolation will be fine.  and at least
% 2 big pixels so that the second downsampling step will be fine.
minextra = max(5,ceil(2*n/sz));

% construct coordinates
if isempty(r)
  r = (1+res)/2;
end
if isempty(c)
  c = (1+res)/2;
end

% figure out the section of the overall image that we will take to be our field-of-view
rowfirst = round(r-sz/2-minextra*(sz/n));
rowlast = round(r+sz/2+minextra*(sz/n));
colfirst = round(c-sz/2-minextra*(sz/n));
collast = round(c+sz/2+minextra*(sz/n));
ressmallrow = rowlast-rowfirst+1;  % number of pixels in the section (row)
ressmallcol = collast-colfirst+1;  % number of pixels in the section (col)

% figure out the final imposed crop
rowfirstreal = round(r-sz/2);
rowlastreal = round(r+sz/2);
colfirstreal = round(c-sz/2);
collastreal = round(c+sz/2);

% figure out total number of pixels to resample the original to (before the final downsampling step)
totalnumrow = ceil(ressmallrow/(sz/n));  % take the ceil to be conservative (so that we're not losing information)
totalnumcol = ceil(ressmallcol/(sz/n));

% figure out indices in the final space
rr = resamplingindices(rowfirst,rowlast,totalnumrow);
cc = resamplingindices(colfirst,collast,totalnumcol);

% convert these indices to the space of the original image
rr2 = (rr - (r-sz/2)) * (n/sz) + 0.5;
cc2 = (cc - (c-sz/2)) * (n/sz) + 0.5;

% the problem is that rr2 and cc2 extend beyond the original image.  so, 
% the original image has to get padded to cover the field-of-view that we're using.
% note that anything outside of [1,n] needs a value (otherwise NaN will result).
numtoaddrowfirst = ceil(1 - rr2(1));
numtoaddrowlast = ceil(rr2(end) - n);
numtoaddcolfirst = ceil(1 - cc2(1));
numtoaddcollast = ceil(cc2(end) - n);
numtopadrow = max([numtoaddrowfirst numtoaddrowlast]);
numtopadcol = max([numtoaddcolfirst numtoaddcollast]);

% interpolate through the original image, then downsample
imB = placematrix(bg*ones(n+2*numtopadrow,n+2*numtopadcol),im);
xx = cc2 + numtopadcol;
yy = rr2 + numtopadrow;
temp = interp2(imB,xx,yy(:),'*cubic');
%figure(1); plot(temp(round(end/2),:)); 
f = imresize(temp,[ressmallrow ressmallcol],'lanczos3');
%figure(2); plot(f(round(end/2),:));
assert(all(isfinite(f(:))));

% crop
f = f(1+(rowfirstreal-rowfirst):end-(rowlast-rowlastreal),1+(colfirstreal-colfirst):end-(collast-collastreal));

% finally, place it in the right spot
f = placematrix(bg*ones(res,res),f,[rowfirstreal colfirstreal]);
