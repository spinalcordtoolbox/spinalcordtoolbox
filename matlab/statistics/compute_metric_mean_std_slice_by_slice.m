function [ mean_std ] = compute_metric_mean_std_slice_by_slice( metric_img, mask_img  )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%   INPUT metric_img: loaded metric image into RPI orientation
%           mask_img: loaded mask image. Default: []
%   OUTPUT mean_std: matrix (number of slices x 2) containing mean value
%   (first position) and STD (second position) of the metric in the mask
%   slice-wise

if exist('mask_img','var')
    nz = size(mask_img,3);
else
    nz = size(metric_img, 3);
    mask_img = ones(dims(metric_img));
end

mean_std = zeros(nz, 2);
for z=1:nz
    slice = metric_img(:, :, z);
    mean_std(z, 1) = mean(slice(logical(mask_img(:,:,z))));
    mean_std(z, 2) = std(slice(logical(mask_img(:,:,z))));

end