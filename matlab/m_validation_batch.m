% Batch for computing Dice coefficient for validation of the registration
% on the template
% The template data registered on the t2_top (output from registration
% pipeline) are compared with the manual segmentation


%--------------------------------------------------------------------------
% Inputs
ext = '.nii.gz';
% Manual segmentation - reference 
file_manual_segmentation = 'manual_segmentation';

% Registered template data into anat (t2_top) space
file_template_levels = 'native_levels_nf_errsm_21_t2';
file_template_WM = 'native_WM_nf_errsm_21_t2';
file_template_GM = 'native_GM_nf_errsm_21_t2';


%--------------------------------------------------------------------------
file_man_seg_resliced = [file_manual_segmentation 'resliced'];
file_levels_binarySC = 'levels_binarySC';
file_template_GWM = 'computed_GWM';
file_template_GWM_treshp5 = 'computed_GWM_thresh_p5';
file_template_GWM_treshp6 = 'computed_GWM_thresh_p6';

% Preprocessing

% Setting all data in the same space 
cmd = ['c3d ' file_manual_segmentation ext ' -interpolation NearestNeighbor -reslice-identity -o ' file_man_seg_resliced ext];
disp(cmd);
[status,result] = unix(cmd);
if(status),error(result);end

% Compute binary SC masks from multilabel template 'levels'
cmd = ['c3d ' file_template_levels ext ' -thresh 3 inf 1 0 -o ' file_levels_binarySC ext];
disp(cmd);
[status,result] = unix(cmd);
if(status),error(result);end

% Compute probabilistic mask of the whole spinal cord
cmd = ['c3d ' file_template_WM ext file_template_GM ext ' -add -o ' file_template_GWM ext];
disp(cmd);
[status,result] = unix(cmd);
if(status),error(result);end

% Thresholding the probabilistic mask - two different thresholds, .5 and .6
cmd = ['c3d ' file_template_GWM ext ' -thresh 0.5 inf 1 0 -o ' file_template_GWM_treshp5 ext];
disp(cmd);
[status,result] = unix(cmd);
if(status),error(result);end

cmd = ['c3d ' file_template_GWM ext ' -thresh 0.6 inf 1 0 -o ' file_template_GWM_treshp6 ext];
disp(cmd);
[status,result] = unix(cmd);
if(status),error(result);end


%--------------------------------------------------------------------------
% Computing Dice coefficient 

% First compute min and max z - part of the image to take into account in
% case one mask is larger than the other one  
manual_seg = read_avw(file_man_seg_resliced);
levels_binarySC = read_avw(file_levels_binarySC);
template_GWM_treshp5 = read_avw(file_template_GWM_treshp5);
template_GWM_treshp6 = read_avw(file_template_GWM_treshp6);

ind_manseg = find(manual_seg);
[mx,my,mz] = ind2sub(size(manual_seg),ind_manseg);
mzmin = min(mz);
mzmax = max(mz);

ind_template = find(levels_binarySC);
[tx,ty,tz] = ind2sub(size(levels_binarySC),ind_template);
tzmin = min(tz);
tzmax = max(tz);

zmin = max(mzmin,tzmin);
zmax = min(mzmin,tzmin);

cmd = ['sct_dice_coefficient ' file_man_seg_resliced ext ' ' levels_binarySC ext ' -b 0 0 ' num2str(zmin) ' -1 -1 ' num2str(zmax)];
disp(cmd);
[status,result] = unix(cmd);
if(status),error(result);end
disp(result);

ind_template = find(template_GWM_treshp5);
[tx,ty,tz] = ind2sub(size(template_GWM_treshp5),ind_template);
tzmin = min(tz);
tzmax = max(tz);

zmin = max(mzmin,tzmin);
zmax = min(mzmin,tzmin);

cmd = ['sct_dice_coefficient ' file_man_seg_resliced ext ' ' file_template_GWM_treshp5 ext ' -b 0 0 ' num2str(zmin) ' -1 -1 ' num2str(zmax)];
disp(cmd);
[status,result] = unix(cmd);
if(status),error(result);end
disp(result);

ind_template = find(template_GWM_treshp6);
[tx,ty,tz] = ind2sub(size(template_GWM_treshp6),ind_template);
tzmin = min(tz);
tzmax = max(tz);

zmin = max(mzmin,tzmin);
zmax = min(mzmin,tzmin);

cmd = ['sct_dice_coefficient ' file_man_seg_resliced ext ' ' file_template_GWM_treshp6 ext ' -b 0 0 ' num2str(zmin) ' -1 -1 ' num2str(zmax)];
disp(cmd);
[status,result] = unix(cmd);
if(status),error(result);end
disp(result);




