% cropping coordinates
xmin = 93;
xmax = 145;
ymin = 95;
ymax = 147;
zmin = 10;
zmax = 34;
% load data
spgr5_nii = load_nii('spgr5.nii.gz');
spgr10_nii = load_nii('spgr10.nii.gz');
spgr20_nii = load_nii('spgr20.nii.gz');
spgr30_nii = load_nii('spgr30.nii.gz');
b1_nii = load_nii('b1_scaling_no_smooth_in_spgr_space.nii.gz');
b1_nii.img = b1_nii.img(xmin:xmax, ymin:ymax, zmin:zmax);


spgr = double(cat(4, spgr5_nii.img(xmin:xmax, ymin:ymax, zmin:zmax), spgr10_nii.img(xmin:xmax, ymin:ymax, zmin:zmax), spgr20_nii.img(xmin:xmax, ymin:ymax, zmin:zmax), spgr30_nii.img(xmin:xmax, ymin:ymax, zmin:zmax)));

mean_spgr_slice10 = squeeze(mean(mean(spgr(:, :, 10, :))))';
mean_b1_slice10 = mean(mean(b1_nii.img(:, :, 10)));

y = mean_spgr_slice10./sin(flipAngles*(pi/180)*mean_b1_slice10);
x = mean_spgr_slice10./tan(flipAngles*(pi/180)*mean_b1_slice10);
[p, S] = polyfit(x, y, 1);
slope = p(1);
intercept = p(2);

if slope>0
    T1 = -tr/log(slope);
else  % due to noise or bad fitting
    T1 = 0;
end
M0 = intercept/(1-exp(-tr/T1));

% assess fitting
figure
hold on
plot(x, y, 'Color', 'blue', 'Marker', '+', 'MarkerSize', 30.0, 'LineStyle','--', 'LineWidth', 2.0)
plot(x, slope*x + intercept, 'Color', 'red', 'Marker', 'x', 'MarkerSize', 30.0, 'LineWidth', 2.0)
hold off
legend('True data', 'Fitting')
grid(gca,'minor') 

