function [ mean_std ] = compute_metric_mean_std_slice_by_slice( metric )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

nz = size(metric, 3);

mean_std = zeros(nz, 2);

for z=1:nz
    slice = metric(:, :, z);
    mean_std(z, 1) = mean(slice(:));
    mean_std(z, 2) = std(slice(:));

end

