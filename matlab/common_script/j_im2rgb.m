% =========================================================================
% FUNCTION
% j_im2rgb.m
%
% Convert an image to RGB format.
%
% INPUTS
% im				(n,m) unsigned. image matrix.
% (clims)			(1,2) float. Min and max value for color scaling (e.g. [20 800])
%
% OUTPUTS
% rgb				(n,m,3) float.
% clims				color scaling.
%
% COMMENTS
% Julien Cohen-Adad 2008-08-02
% =========================================================================
function [rgb clims] = j_im2rgb(im,clims)


% initialization
if (nargin<1) help j_im2rgb, return; end
if (nargin<2) clims = ''; end

% get image size
[nx ny] = size(im);

% truncate colormap
if ~isempty(clims)
	% truncate low values
	im = im + (im<clims(1)).*(clims(1)-im);
	% truncate high values
	im = im + (im>clims(2)).*(clims(2)-im);
end

% retrieve min and max
clims(1) = min(im(:));
clims(2) = max(im(:));

% scale image from 0 to 1
im = (im - clims(1))./clims(2);

% scale image from 1 to 64
im = 1 + round(63*im);

% read the colormap matrix
figure('visible','off')
gray_cmap = colormap(gray);
close

% create RGB image
rgb = zeros(nx,ny,3);
for i_rgb = 1:3,
	vals_for_im = gray_cmap(im,i_rgb);
	rgb(:,:,i_rgb) = reshape(vals_for_im,nx,ny);
end



% % check for NaN
% temp = reshape(im,1,nx*ny);
% ind_nan = find(isnan(im));
% temp(ind_nan) = 0;
% im = reshape(temp,nx,ny);