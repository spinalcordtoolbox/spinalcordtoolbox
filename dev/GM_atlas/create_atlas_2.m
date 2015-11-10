
%-------------- Grey matter tracts template construction -----------------
%--------------------------------------------------------------------------

% This script is made to construct a partial volume grey
% matter tracts template, using raw anatomic atlas information which
% contains the grey matter tracts and a spinal cord template (or data) on
% which we want to integrate information on the grey matter tracts

% v8: fast version with one registration per vertebral level
%       interpolation between slice to smooth in z direction
%       full symmetrization

%----------------------------- Dependencies -------------------------------
% Matlab dependencies:
% - image processing toolbox functions
% - m_normalize.m : normalization function
% - dnsamplelin.m : function for downsampling by computing mean value for
%   each region
% - m_linear_interp.m
%
% Other dependencies: FSL, sct_c3d, ANTs

dbstop if error

%------------------------------- Inputs -----------------------------------
% file_template: the template slice we want to show tracts on
% file_atlas: the raw atlas information containing labeled regions 
% file_mask: a binary version of the altas with a white matter mask
% ext_altas: extension for the image file of the altas and its mask
% 
% num_slice_ref: the number of the reference slice which will be used for
%   direct registration of the atlas
% interp_factor: the interpolation factor appropriate for getting the
%   template close to the atlas slice
% label_values: vector containing the list of label values in the atlas,
%   which should be integers in range [0,255]
%addpath(genpath('~/code/'));

% get path of FSL
[status, path_fsl] = unix('echo $FSLDIR');
% get FSL matlab functions
path_fsl_matlab = strcat(path_fsl, '/etc/matlab');
% add to path
addpath(path_fsl_matlab);
% get path of the toolbox
[status, path_sct] = unix('echo $SCT_DIR');
% define path of template
path_template = strcat(path_sct, '/data/template/');
% name of the WM template. Default is 'MNI-Poly-AMU_GM'
file_template = 'MNI-Poly-AMU_GM';
% path to the image file that contains the drawing of the GM atlas.
path_atlas_data = strcat('/Users/tamag/Desktop/GM_atlas/def_new_atlas/test_2/');
% file name of the full atlas
file_atlas = 'greyscale_antisym_resampled_registered_crop_resized_transpose';
% file name of the binary mask that helps for the registration to the MNI-Poly-AMU
file_mask = 'gm_white_resampled_registered_crop_resized_transpose';
ext_atlas = '.png';

% corresponds to mid-C4 in the MNI-Poly-AMU template
z_slice_ref = 324;
% interpolation factor for the MNI-Poly-AMU template in order to match the hi-res grays atlas
interp_factor = 6;

% values of the label in the atlas file (file_atlas). Each value corresponds to a given tract, e.g., corticospinal left.
% NB: 255=WM
% label_values = [14 26 38 47 52 62 70 82 89 94 101 107 112 116 121 146 152 159 167 173 180 187 194 199 204 208 214 219 224 230 238 255];
label_left = [44 80 120];
label_right = [150 190 220];
label_pve = [255];
label_values = [label_left, label_right, label_pve];
% these are the value corresponding to the slice number (z) on the MNI-Poly-AMU template, at which the atlas will be warped. It corresponds to the mid-levels as well as the level of the intervertebral disks.
% NB: to extract these values, you have to look at the T2 and WM template, because this script will crop the WM template (which can be smaller than the T2), therefore the maximum z cannot exceed the zmax that will be generated in the cropped version of the WM template.
z_disks_mid = [483 476 466 455 440 423 406 387 371 356 339 324 303 286 268 248 229 208 186 166 143 122 98 79 53 35 13 0];
% same as before-- except that C6 mid-vertebral is not listed. 
z_disks_mid_noC6 = [483 476 466 455 440 423 406 387 371 356 339 303 286 268 248 229 208 186 166 143 122 98 79 53 35 13 0];


%--------------------------------------------------------------------------
%----------------------- Starting the pipeline -------------------------
%--------------------------------------------------------------------------

ext = '.nii.gz';
fsloutputype = 'export FSLOUTPUTTYPE=NIFTI_GZ; ';
path_out = 'GMtracts_outputs/';
path_ctrl = [path_out 'registered_template/'];
path_results = [path_out 'final_results/'];
prefix_out = 'GMtract_';

template_mask = [path_out file_template '_mask'];
template_cropped = [path_out file_template '_c6v'];
template_cropped_interp = [template_cropped '_int' num2str(interp_factor)];
template_cig = [template_cropped '_intgauss' num2str(interp_factor)];
templateci_slice_ref = [template_cropped_interp '_slice_ref'];
templateci_slice_ref_thresh = [templateci_slice_ref '_thresh'];
templateci_sr_nohd = [templateci_slice_ref '_nohd'];
templateci_thresh = [template_cropped_interp '_thrp6'];
templateci_srt_nohd = [templateci_slice_ref_thresh '_nohd'];
templatecit_slice_ref = [templateci_thresh '_slice_ref'];

tracts_atlas = cell(1,length(label_values));
mask_nohd = [path_out file_mask];
atlas_nifti = [path_out file_atlas];

num_slice_ref = z_slice_ref + 1;
perc_up = 100*interp_factor;
perc_dn = 100/interp_factor;
prefix_ants = [path_out 'reg_'];
prefix_ants_ref = [path_out 'reg_ref_'];
affine_atlas = 'Affine0GenericAffine.mat';  %[prefix_ants_ref 'Affine.txt'];
Warp_atlas = [prefix_ants 'Warp_init' ext];
Warp_tmp = [prefix_ants 'Warp_init'];
suffix_ants = '_reg';


% if folder exists, delete it
if exist(path_out)
    m = input('Output folder already exists. Delete it? (y/n) ', 's');
    if m == 'y'
        cmd = ['rm -rf ' path_out];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
        mkdir(path_out);
        mkdir(path_ctrl);
        mkdir(path_results);
    end
else
    mkdir(path_out);
    mkdir(path_ctrl);
    mkdir(path_results);
end


%--------------------------------------------------------------------------
%--- Preliminary operations: cropping and interpolation of the template ---

% Thresholding the template
cmd = ['isct_c3d ' path_template file_template ext ' -threshold -inf 0.001 0 1 -o ' template_mask ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

cmd = ['isct_c3d ' path_template file_template ext ' ' template_mask ext ' -multiply -o ' template_mask ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Cropping the template
cmd = ['isct_c3d ' template_mask ext ' -trim 6vox -o ' template_cropped ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Interpolation of the template
cmd = ['isct_c3d ' template_cropped ext ' -interpolation Linear -resample ',...
    num2str(perc_up) 'x' num2str(perc_up) 'x100% -o ' template_cropped_interp ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Extract reference slice
cmd = ['isct_c3d ' template_cropped_interp ext ' -slice z ' num2str(z_slice_ref) ' -o ' templateci_slice_ref ext];
% cmd = [fsloutputype 'fslroi ' template_cropped_interp ' ' templateci_slice_ref ' 0 -1 0 -1 ' num2str(z_slice_ref) ' 1 '];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
% change field dim0 from 3 to 2
cmd = [fsloutputype 'fslroi ' templateci_slice_ref ' ' templateci_slice_ref ' 0 -1 0 -1 0 -1'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% % remove geometrical information -- WHY?
% [sliceref,~,scales] = read_avw(templateci_slice_ref);
% sliceref = m_normalize(sliceref);
% save_avw(sliceref,templateci_sr_nohd,'d',scales);

% Binarization of the reference slice for the registration of the atlas
cmd = ['isct_c3d ' templateci_slice_ref ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_slice_ref_thresh ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
% change field dim0 from 3 to 2
cmd = [fsloutputype 'fslroi ' templateci_slice_ref_thresh ' ' templateci_slice_ref_thresh ' 0 -1 0 -1 0 -1'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% Binarization of the template for slice coregistration
cmd = ['isct_c3d ' template_cropped_interp ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_thresh ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
% change field dim0 from 3 to 2
cmd = [fsloutputype 'fslroi ' templateci_thresh ' ' templateci_thresh ' 0 -1 0 -1 0 -1'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% % Get a version of binarized ref slice without geometrical information.-- WHY???
% [sliceref,~,scales] = read_avw(templateci_slice_ref_thresh);
% sliceref = m_normalize(sliceref);
% save_avw(sliceref,templateci_srt_nohd,'d',scales);



% Save the atlas and mask into a nifti with the same scales as the template
[slice_ref,~,scales] = read_avw(templateci_slice_ref_thresh);
% [slice_ref,~,scales] = read_avw(templateci_srt_nohd);
atlas = imread([path_atlas_data file_atlas ext_atlas]);
% create one file for each tract
for label = 1:length(label_values)
    temp = zeros(size(atlas));
    ind = find( atlas == label_values(label) );
    temp(ind) = 1;
    tracts_atlas{label} = temp;
    tract_atlas = [path_out 'tract_atlas_' num2str(label)];
    save_avw(tracts_atlas{label},tract_atlas,'d',scales);
    % copy header info template --> tract_atlas
    cmd = ['fslcpgeom ' templateci_slice_ref_thresh ' ' tract_atlas ' -d '];
    disp(cmd); [status,result]=unix(cmd); if(status), error(result); end
end

% Normalizes atlas between [0,1]
atlas = m_normalize(atlas);
save_avw(atlas,atlas_nifti,'d',scales);
% Normalizes binary version of atlas between [0,1]
mask = imread([path_atlas_data file_mask ext_atlas]);
mask = m_normalize(mask);
save_avw(mask,mask_nohd,'d',scales);

% copy header info template --> mask
cmd = ['fslcpgeom ' templateci_slice_ref_thresh ' ' mask_nohd ' -d '];
disp(cmd); [status,result]=unix(cmd); if(status), error(result); end
% copy header info template --> atlas
cmd = ['fslcpgeom ' templateci_slice_ref_thresh ' ' atlas_nifti ' -d '];
disp(cmd); [status,result]=unix(cmd); if(status), error(result); end

% Initializing outputs
[templateCROP,dimsCROP,scalesCROP] = read_avw(template_cropped);
[templateCI,dimsCI,scalesCI] = read_avw(template_cropped_interp);

tractsHR = cell(1,length(label_values));
tracts = cell(1,length(label_values));

for label = 1:length(label_values)
    tractsHR{label} = zeros(size(templateCI));
    tracts{label} = zeros(size(templateCROP));
end



%--------------------------------------------------------------------------
% Initial registration of the atlas to the reference slice of the template


% estimate affine transformation from atlas to template
% here, using flag -r 0 to initialize transformations based on geometrical center of images
cmd = ['isct_antsRegistration --dimensionality 2 -m MeanSquares[' templateci_slice_ref_thresh ext ',' mask_nohd ext ',1,4] -t Affine[1] --convergence 100x10 -s 1x0 -f 2x1 -r [' templateci_slice_ref_thresh ext ',' mask_nohd ext ', 0] -o [' path_out 'Affine,' mask_nohd '_affine' ext ']'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)

% estimate diffeomorphic transformation
cmd =['isct_antsRegistration --dimensionality 2 --initial-moving-transform ' path_out affine_atlas ' ',...
    '--transform SyN[0.1,3,0] --metric MeanSquares[' templateci_slice_ref_thresh ext ',' mask_nohd ext ',1,4] ',... 
    '--convergence 100x10 --shrink-factors 4x1 --smoothing-sigmas 0x0mm ',...
    '--output [' prefix_ants_ref ',' mask_nohd '_affine_warp' ext '] --collapse-output-transforms 1'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)

% Rename warping field
movefile([prefix_ants_ref '1Warp.nii.gz'],[Warp_tmp ext]);

% Constraint the warping field to preserve symmetry
cmd = ['isct_c3d -mcs ' Warp_tmp ext ' -oo ' Warp_tmp 'x' ext ' ' Warp_tmp 'y' ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

[warpx,dims,scales] = read_avw([Warp_tmp 'x' ext]);
warpy = read_avw([Warp_tmp 'y' ext]);
warpx = (warpx - warpx(end:-1:1,:)) / 2;
warpy = (warpy + warpy(end:-1:1,:)) / 2;

save_avw(warpx,[Warp_tmp 'xa' ext],'d',scales);
save_avw(warpy,[Warp_tmp 'ys' ext],'d',scales);

cmd = ['isct_c3d ' Warp_tmp 'x' ext ' ' Warp_tmp 'xa' ext ' -copy-transform -o ' Warp_tmp 'xa' ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

cmd = ['isct_c3d ' Warp_tmp 'y' ext ' ' Warp_tmp 'ys' ext ' -copy-transform -o ' Warp_tmp 'ys' ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

% Applying tranform to the mask
cmd = ['isct_antsApplyTransforms -d 2 -i ' mask_nohd ext ' -o ' mask_nohd suffix_ants ext ' -t ' Warp_atlas ' ' path_out affine_atlas ' -r ' templateci_slice_ref_thresh ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

% Applying tranform to the initial atlas
cmd = ['isct_antsApplyTransforms -d 2 -i ' atlas_nifti ext ' -o ' atlas_nifti suffix_ants ext ' -t ' Warp_atlas ' ' path_out affine_atlas ' -r ' templateci_slice_ref_thresh ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

% Applying tranform to the tract files and copying geometry and saving
for label = 1:length(label_left)
    label_l = label;
    label_r = label+length(label_left);
    tract_atlas_g = [path_out 'tract_atlas_' num2str(label_l)];
    tract_atlas_d = [path_out 'tract_atlas_' num2str(label_r)];
    
    cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_g ext ' -o ' tract_atlas_g suffix_ants ext ' -t ' Warp_atlas ' ' path_out affine_atlas ' -r ' templateci_slice_ref_thresh ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    
    cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_d ext ' -o ' tract_atlas_d suffix_ants ext ' -t ' Warp_atlas ' ' path_out affine_atlas ' -r ' templateci_slice_ref_thresh ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    
    tract_reg_g = [path_out 'tract_atlas_' num2str(label_l) suffix_ants];
    temp_g = read_avw(tract_reg_g);
    
    tract_reg_d = [path_out 'tract_atlas_' num2str(label_r) suffix_ants];
    temp_d = read_avw(tract_reg_d);
    
    % Replace isolated values with the mean of the adjacent values
    for i = 2:size(temp_g,1)-1
        for j = 2:size(temp_g,2)-1
            test = (temp_g(i,j)==temp_g(i-1,j)) || (temp_g(i,j)==temp_g(i,j-1)) || (temp_g(i,j)==temp_g(i+1,j)) || (temp_g(i,j)==temp_g(i,j+1));
            if(~test)
                temp_g(i,j) = (temp_g(i-1,j) + temp_g(i+1,j) + temp_g(i,j+1) + temp_g(i,j-1))/4;
            end
        end
    end
    
    for i = 2:size(temp_d,1)-1
        for j = 2:size(temp_d,2)-1
            test = (temp_d(i,j)==temp_d(i-1,j)) || (temp_d(i,j)==temp_d(i,j-1)) || (temp_d(i,j)==temp_d(i+1,j)) || (temp_d(i,j)==temp_d(i,j+1));
            if(~test)
                temp_d(i,j) = (temp_d(i-1,j) + temp_d(i+1,j) + temp_d(i,j+1) + temp_d(i,j-1))/4;
            end
        end
    end
    
    % Symmetry constraint for left and right tracts
    temp_sum = temp_g + temp_d;
    temp_sum_flip = temp_sum(end:-1:1,:);
    temp_sym = (temp_sum + temp_sum_flip) / 2;
    
    temp_g(1:end/2,:) = 0;
    temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
    temp_d(1:end/2,:) = temp_sym(1:end/2,:);
    temp_d(1+end/2:end,:) = 0;
    
    tractsHR{label_l}(:,:,num_slice_ref) = temp_g;
    tractsHR{label_r}(:,:,num_slice_ref) = temp_d;
end

% Apply tranform to the PVE tract files
for label = length([label_left, label_right])+1:length(label_values)
    tract_atlas = [path_out 'tract_atlas_' num2str(label)];
    
    cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas ext ' -o ' tract_atlas suffix_ants ext ' -t ' Warp_atlas ' ' path_out affine_atlas ' -r ' templateci_slice_ref_thresh ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    
    tract_reg_g = [path_out 'tract_atlas_' num2str(label) suffix_ants];
    temp_g = read_avw(tract_reg_g);
    
    % Replace isolated values with the mean of the adjacent values
    for i = 2:size(temp_g,1)-1
        for j = 2:size(temp_g,2)-1
            test = (temp_g(i,j)==temp_g(i-1,j)) || (temp_g(i,j)==temp_g(i,j-1)) || (temp_g(i,j)==temp_g(i+1,j)) || (temp_g(i,j)==temp_g(i,j+1));
            if(~test)
                temp_g(i,j) = (temp_g(i-1,j) + temp_g(i+1,j) + temp_g(i,j+1) + temp_g(i,j-1))/4;
            end
        end
    end
    
    tractsHR{label}(:,:,num_slice_ref) = temp_g;
end


%--------------------------------------------------------------------------
%---------------------- Construction of the template ----------------------



cmd = ['isct_c3d ' templateci_thresh ext ' -slice z ' num2str(z_slice_ref) ' -o ' templatecit_slice_ref ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

[mstatus,msg] = copyfile([templateci_slice_ref ext],[template_cropped_interp '_slice' z_slice_ref ext]);
if(~mstatus), error(msg); end

reference_slice = [template_cropped_interp '_slice' z_slice_ref];
%thr_ref_slice = templatecit_slice_ref;



nb_slices = length(z_disks_mid);

disp('*** Register slice i+1 to slice i ***')
% loop across selected slices for registration. Stops at n-1, because it
% registers slice i onto slice i+1.
for iz = 1:nb_slices-1
    
    zslice = z_disks_mid(iz);
    zslicenext = z_disks_mid(iz+1);

    templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    templatecit_slicenext = [templateci_thresh '_slice' num2str(zslicenext)];
    warp_slice = [prefix_ants '1Warp.nii.gz'];
    affine_slice = [prefix_ants 'Affine.txt'];
    warp_temp = [prefix_ants '0Warp'];
    
    % extract slice corresponding to z=zslice
    cmd = ['isct_c3d ' templateci_thresh ext ' -slice z ' num2str(zslice) ' -o ' templatecit_slice ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

    % extract slice corresponding to z=zslice+1 (for neighrest-slice registration)
    cmd = ['isct_c3d ' templateci_thresh ext ' -slice z ' num2str(zslicenext) ' -o ' templatecit_slicenext ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

    % registration of slice i+1 on i
    % forward warping field is: reg_zslice_0Warp.nii.gz
    % backward warping field is: reg_zslice_0InverseWarp.nii.gz
%     cmd =['isct_antsRegistration --dimensionality 2 ',...
%         '--transform SyN[0.2,3,0] --metric CC[' templatecit_slice ext ',' templatecit_slicenext ext ',1,4] ',... 
%         '--convergence 200x20 --shrink-factors 2x1 --smoothing-sigmas 7x7vox ',...
%         '--output [' [prefix_ants num2str(zslice) '_'] ',' prefix_ants 'slicenext_to_slice.nii.gz]'];
    % WARNING!!! (2014-08-04) the binary version of isct_antsRegistration from
    % ANTs website crashes when using option --transform BSplineSyN. Use
    % the version from github instead.
    cmd =['isct_antsRegistration --dimensionality 2 ',...
        '--transform BSplineSyN[0.2,3] --metric MeanSquares[' templatecit_slice ext ',' templatecit_slicenext ext ',1,4] ',... 
        '--convergence 100x5 --shrink-factors 2x1 --smoothing-sigmas 0x0vox ',...
        '--output [' [prefix_ants num2str(zslicenext) '_'] ',' prefix_ants 'slicenext_to_slice.nii.gz]'];    
%     cmd =['isct_antsRegistration --dimensionality 2 ',...
%         '--transform BSplineSyN[0.2,3] --metric MeanSquares[' templatecit_slice ext ',' templatecit_slicenext ext ',1,4] ',... 
%         '--convergence 100x20 --shrink-factors 2x1 --smoothing-sigmas 0x0vox ',...
%         '--output [' [prefix_ants num2str(zslice) '_'] ',' prefix_ants 'slicenext_to_slice.nii.gz]'];    
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
end

disp('*** Concatenate warping fields for each slice ***')
% at this point
% reg_476_0Warp.nii.gz is the warping field for 476 --> 483
% reg_476_0InverseWarp.nii.gz is the warping field for 483 --> 476
% the goal now, is to concatenate warping fields, to obtain "reg_concat_483": 387 (ref) --> 483 
% and so on...
% so the concatenation should be done like that:
% for zslice > zref:
%   reg_concat_483 = reg_476_0Warp + ... + reg_406_0Warp + reg_387_0Warp
% for zslice < zref:
%   reg_concat_356 = reg_371_0InverseWarp + reg_356_0InverseWarp
% find index for zslice ref
izref = find(z_disks_mid == z_slice_ref);
nb_slices = length(z_disks_mid_noC6);
for iz = 1:nb_slices
    
%     templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    zslice = z_disks_mid_noC6(iz);
    templatecit_slice = [templateci_thresh '_slice' num2str(z_disks_mid(izref))];
   
    % output concatenated field is: reg_concat_"zslice"
    cmd = ['isct_ComposeMultiTransform 2 ', prefix_ants, 'concat_', num2str(zslice), ext, ' -R ', templatecit_slice, ext];

    if zslice > z_slice_ref
        % if zslice is superior to ref slice, then concatenate forward warping
        % fields, from the zref to zslice.
        % concatenated warping field: warp_temp = reg_0Warp
        for izconcat = iz:izref-1
            cmd = [cmd, ' ', prefix_ants num2str(z_disks_mid(izconcat+1)), '_0Warp', ext];
        end
    else
        % if zslice is inferior to ref slice, then concatenate backward warping
        % fields, from the zref to zslice.
        for izconcat = izref:iz
            cmd = [cmd, ' ', prefix_ants num2str(z_disks_mid(izconcat+1)), '_0InverseWarp', ext];
        end
    end
    disp(cmd)
    [status, result] = unix(cmd);
end


disp('*** Adjust warping fields to minimize the propagation of error from concatenation ***')
nb_slices = length(z_disks_mid_noC6);
for iz = 1:nb_slices
    
    zslice = z_disks_mid_noC6(iz);    
    templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    warp_slice = [ prefix_ants, 'concat_', num2str(zslice), ext];
    izref = find(z_disks_mid == z_slice_ref);
    
    % register refslice to zslice using concatenated warping field as
    % initial transformation
    cmd =['isct_antsRegistration --dimensionality 2 ',...
        '--initial-moving-transform ', warp_slice, ' ',...
        '--transform BSplineSyN[0.2,3] --metric MeanSquares[' templatecit_slice ext ',' templatecit_slice_ref ext ',1,4] ',... 
        '--convergence 200x5 --shrink-factors 2x1 --smoothing-sigmas 0x0vox ',...
        '--output [' prefix_ants, 'concat_', num2str(zslice) ',' templatecit_slice_ref 'to_' num2str(zslice) ext ']' ];
    disp(cmd)
    [status,result] = unix(cmd);
%     disp(result)
    if(status),error(result);end 
    
    % Replace the concatenated warping field with the new warping field 
    movefile([prefix_ants, 'concat_', num2str(zslice), '0Warp', ext], warp_slice)
    
end


disp('*** Symmetrize warping fields ***')
nb_slices = length(z_disks_mid_noC6);
for iz = 1:nb_slices
    
    zslice = z_disks_mid_noC6(iz);
    
    warp_temp = [ prefix_ants, 'concat_', num2str(zslice)];
    warp_slice = [ prefix_ants, 'concat_sym_', num2str(zslice), ext];

    % Constraint the warping field to preserve symmetry
    cmd = ['isct_c3d -mcs ' warp_temp ext ' -oo ' warp_temp 'x' ext ' ' warp_temp 'y' ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    
    % read warping field
    [warpx,dims,scales] = read_avw([warp_temp 'x' ext]);
    warpy = read_avw([warp_temp 'y' ext]);
    warpx = (warpx - warpx(end:-1:1,:)) / 2;
    warpy = (warpy + warpy(end:-1:1,:)) / 2;
    
    save_avw(warpx,[warp_temp 'xa' ext],'d',scales);
    save_avw(warpy,[warp_temp 'ys' ext],'d',scales);
    
    cmd = ['isct_c3d ' warp_temp 'x' ext ' ' warp_temp 'xa' ext ' -copy-transform -o ' warp_temp 'xa' ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    
    cmd = ['isct_c3d ' warp_temp 'y' ext ' ' warp_temp 'ys' ext ' -copy-transform -o ' warp_temp 'ys' ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    
    cmd = ['isct_c3d ' warp_temp 'xa' ext ' ' warp_temp 'ys' ext ' -omc 2 ' warp_slice];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
end


disp('*** Applies warping fields for registration to each intermediate slice ***')
nb_slices = length(z_disks_mid_noC6)
for iz = 1:nb_slices
    
    disp(['SLICE #: ', num2str(iz), '/', num2str(nb_slices)])
    
    zslice = z_disks_mid_noC6(iz);
    numSlice = zslice+1;
    templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    atlas_slice = [atlas_nifti suffix_ants num2str(zslice)];
    warp_slice = [ prefix_ants, 'concat_sym_', num2str(zslice), ext];
    
    % Apply transform to reference slice as a control
    cmd = ['isct_antsApplyTransforms -d 2 -i ' templatecit_slice_ref ext ' -o ' templatecit_slice_ref suffix_ants num2str(zslice) ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
    
    % Apply transform to the initial atlas as a control
%     cmd = ['sct_WarpImageMultiTransform 2 ' atlas_nifti ext ' ' atlas_slice ext ' ' warp_slice ' ' Warp_atlas ' ' path_out affine_atlas ' -R ' templatecit_slice ext];
    cmd = ['isct_antsApplyTransforms -d 2 -i ' atlas_nifti ext ' -o ' atlas_slice ext ' -t ' Warp_atlas ' ' path_out affine_atlas ' -r ' templateci_slice_ref_thresh ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
    cmd = ['isct_c3d ' templatecit_slice_ref ' ' atlas_slice ext ' -copy-transform -o ' atlas_slice ext];  % copy geom-- added: 2014-08-30
    disp(cmd); [status,result]=unix(cmd); if(status), error(result); end, %disp(result)
    cmd = ['isct_antsApplyTransforms -d 2 -i ' atlas_slice ext ' -o ' atlas_slice suffix_ants ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
    
    % Apply tranform to the GM tract files and constraint to be symmetric
    for label = 1:length(label_left)
        label_l = label;
        label_r = label+length(label_left);
        tract_atlas_g = [path_out 'tract_atlas_' num2str(label_l)];
        tract_atlas_d = [path_out 'tract_atlas_' num2str(label_r)];

        % LEFT
        cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_g ext ' -o ' tract_atlas_g suffix_ants ext ' -t ' Warp_atlas ' ' path_out affine_atlas ' -r ' templateci_slice_ref_thresh ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
        cmd = ['isct_c3d ' templatecit_slice_ref ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext ext];  % copy geom-- added: 2014-08-30
        disp(cmd); [status,result]=unix(cmd); if(status), error(result); end, %disp(result)
        cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_g suffix_ants ext ' -o ' tract_atlas_g suffix_ants ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
        
        % RIGHT
        cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_d ext ' -o ' tract_atlas_d suffix_ants ext ' -t ' Warp_atlas ' ' path_out affine_atlas ' -r ' templateci_slice_ref_thresh ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
        cmd = ['isct_c3d ' templatecit_slice_ref ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext ext];  % copy geom-- added: 2014-08-30
        disp(cmd); [status,result]=unix(cmd); if(status), error(result); end, %disp(result)
        cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_d suffix_ants ext ' -o ' tract_atlas_d suffix_ants ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

        % copy header from template to registered atlas
        % NB: changed templateci_slice to templatecit_slice (2014-08-04)
        cmd = ['isct_c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
        
        cmd = ['isct_c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
        
        tract_reg_g = [path_out 'tract_atlas_' num2str(label_l) suffix_ants];
        temp_g = read_avw(tract_reg_g);
        
        tract_reg_d = [path_out 'tract_atlas_' num2str(label_r) suffix_ants];
        temp_d = read_avw(tract_reg_d);
        
        % Replace isolated values with the mean of the adjacent values
        for i = 2:size(temp_g,1)-1
            for j = 2:size(temp_g,2)-1
                test = (temp_g(i,j)==temp_g(i-1,j)) || (temp_g(i,j)==temp_g(i,j-1)) || (temp_g(i,j)==temp_g(i+1,j)) || (temp_g(i,j)==temp_g(i,j+1));
                if(~test)
                    temp_g(i,j) = (temp_g(i-1,j) + temp_g(i+1,j) + temp_g(i,j+1) + temp_g(i,j-1))/4;
                end
            end
        end
        
        for i = 2:size(temp_d,1)-1
            for j = 2:size(temp_d,2)-1
                test = (temp_d(i,j)==temp_d(i-1,j)) || (temp_d(i,j)==temp_d(i,j-1)) || (temp_d(i,j)==temp_d(i+1,j)) || (temp_d(i,j)==temp_d(i,j+1));
                if(~test)
                    temp_d(i,j) = (temp_d(i-1,j) + temp_d(i+1,j) + temp_d(i,j+1) + temp_d(i,j-1))/4;
                end
            end
        end
        
        % Symmetry constraint for left and right tracts
        temp_sum = temp_g + temp_d;
        temp_sum_flip = temp_sum(end:-1:1,:);
        temp_sym = (temp_sum + temp_sum_flip) / 2;
        
        temp_g(1:end/2,:) = 0;
        temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
        temp_d(1:end/2,:) = temp_sym(1:end/2,:);
        temp_d(1+end/2:end,:) = 0;
        
        tractsHR{label_l}(:,:,numSlice) = temp_g;
        tractsHR{label_r}(:,:,numSlice) = temp_d;
        
    end

    % Apply tranform to the PVE tract files
    for label = length([label_left, label_right])+1:length(label_values)
        tract_atlas = [path_out 'tract_atlas_' num2str(label)];

        cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas ext ' -o ' tract_atlas suffix_ants ext ' -t ' Warp_atlas ' ' path_out affine_atlas ' -r ' templateci_slice_ref_thresh ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
        cmd = ['isct_c3d ' templatecit_slice_ref ' ' tract_atlas suffix_ants ext ' -copy-transform -o ' tract_atlas suffix_ants ext ext];  % copy geom-- added: 2014-08-30
        disp(cmd); [status,result]=unix(cmd); if(status), error(result); end, %disp(result)
        cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas suffix_ants ext ' -o ' tract_atlas suffix_ants ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
        
        % copy header from template to registered atlas
        % NB: changed templateci_slice to templatecit_slice (2014-08-04)
        cmd = ['isct_c3d ' templatecit_slice ext ' ' tract_atlas suffix_ants ext ' -copy-transform -o ' tract_atlas suffix_ants ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
                
        tract_reg = [path_out 'tract_atlas_' num2str(label) suffix_ants];
        temp_g = read_avw(tract_reg);
                
        % Replace isolated values with the mean of the adjacent values
        for i = 2:size(temp_g,1)-1
            for j = 2:size(temp_g,2)-1
                test = (temp_g(i,j)==temp_g(i-1,j)) || (temp_g(i,j)==temp_g(i,j-1)) || (temp_g(i,j)==temp_g(i+1,j)) || (temp_g(i,j)==temp_g(i,j+1));
                if(~test)
                    temp_g(i,j) = (temp_g(i-1,j) + temp_g(i+1,j) + temp_g(i,j+1) + temp_g(i,j-1))/4;
                end
            end
        end
        
        % Symmetry constraint for left and right tracts
        tractsHR{label}(:,:,numSlice) = temp_g;
    end    
   
    % Move control files to control folder
    reg_slice_current = [templatecit_slice_ref suffix_ants num2str(zslice)];
    movefile([reg_slice_current ext],path_ctrl);
    movefile([atlas_slice suffix_ants ext],path_ctrl);

end


%----------------- Interpolation between computed slices ------------------

for label = 1:length(label_values)
    for k = 1:length(z_disks_mid)-1
        tractsHR{label} = m_linear_interp(tractsHR{label},z_disks_mid(k)+1,z_disks_mid(k+1)+1);
    end
end



%-------------- Downsampling and partial volume computation ---------------
max_indx = max(z_disks_mid(:));


for label = 1:length(label_values)
    for zslice = 0:max_indx
        numSlice = zslice+1;
        tracts{label}(:,:,numSlice) = dnsamplelin(tractsHR{label}(:,:,numSlice),interp_factor);
    end
end



%--- Loop on labels to compute partial volume values without HR version ---

% create variable of tract numbering with 2 digits starting at 00
cell_tract = m_numbering(length(label_values), 2, 0);
% loop across tracts
for label = 1:length(label_values)
    
    % Save ML version and copy geometry
    filetractML = [path_results prefix_out '_' cell_tract{label}];
    save_avw(tracts{label},filetractML,'d',scalesCROP);
    cmd = ['isct_c3d ' template_cropped ext ' ' filetractML ext ' -copy-transform -o ' filetractML ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
    
	 % Reslice into native template space
	cmd = ['isct_c3d ' path_template file_template ext ' ' filetractML ext ' -reslice-identity -o ' filetractML ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(resul)

    % copy geometry from white matter template
    cmd = ['fslcpgeom ', path_template, file_template, ' ', filetractML];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
end


% FINISHED!
bricon = ' -b 0.2,1 '
disp 'Done! To see results, type:'
disp(['cd ',path_results])
disp(['fslview ',path_template,'MNI-Poly-AMU_T2.nii.gz -b 0,5000 ',path_template,'MNI-Poly-AMU_GM.nii.gz -b 0.2,1 GMtract__00.nii.gz -l Red',bricon,'GMtract__01.nii.gz -l Green',bricon,'GMtract__02.nii.gz -l Blue',bricon,'GMtract__03.nii.gz -l Yellow',bricon,'GMtract__04.nii.gz -l Pink ',bricon,'GMtract__05.nii.gz -l Cool ',bricon,'GMtract__06.nii.gz -l Copper ',bricon,' &'])


