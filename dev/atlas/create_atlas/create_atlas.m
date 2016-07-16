
%-------------- White matter tracts template construction -----------------
%--------------------------------------------------------------------------

% This script is made to construct a partial volume white
% matter tracts template, using raw anatomic atlas information which
% contains the white matter tracts and a spinal cord template (or data) on
% which we want to intergrate information on the white matter tracts

%----------------------------- Dependencies -------------------------------
% Matlab dependencies:
% - image processing toolbox functions
% - m_normalize.m : normalization function
% - dnsamplelin.m : function for downsampling by computing mean value for
%   each region
% - m_linear_interp.m
%
% Other dependencies: 
% - FSL
% - c3d
% - ANTs

dbstop if error

% get path of running script
[path_script, b, c] = fileparts(which('create_atlas.m'));
% go to this path
cd(path_script)

% ADD TO PATH
% addpath(genpath('~'));
addpath(pwd);

%------------------------------- Inputs -----------------------------------
% file_template: the template slice we want to show tracts on
% file_atlas: the raw atlas information containing labeled regions 
% file_mask: a binary version of the altas with a white matter mask
% ext_atlas: extension for the image file of the altas and its mask
% 
% num_slice_ref: the number of the reference slice which will be used for
%   direct registration of the atlas
% interp_factor: the interpolation factor appropriate for getting the
%   template close to the atlas slice
% label_values: vector containing the list of label values in the atlas,
%   which should be integers in range [0,255]

% Absolute path of output results (add "/" at the end)
path_out = '/Users/julien/code/spinalcordtoolbox/dev/atlas/create_atlas/results/';
% get path of FSL
[status, path_fsl] = unix('echo $FSLDIR');
% get FSL matlab functions
path_fsl_matlab = strcat(path_fsl, '/etc/matlab');
% add to path
addpath(path_fsl_matlab);
% define SCT path
path_sct = '/Users/julien/code/spinalcordtoolbox/';
% define path of template
path_template = '/Users/julien/data/PAM50/template/';
% name of the WM template. Don't put extension.
file_template = 'PAM50_wm';  % PAM50_WM

%--------------------------------------------------------------------------
% LABEL PARAMETERS
%--------------------------------------------------------------------------
% path to the image file that contains the drawing of the WM atlas from Grays anatomy.
path_atlas_data = strcat(path_sct, 'dev/atlas/raw_data/');
% file name of the full atlas
file_atlas = 'atlas_grays_cerv_sym_correc_r6';
% file name of the binary mask that helps for the registration to the MNI-Poly-AMU
file_mask = 'mask_grays_cerv_sym_correc_r5';
ext_atlas = '.png';
% text file linking label values, ID and description
file_atlas_txt = 'atlas_grays_cerv_sym_correc_r6_label.txt';
% values of the label in the atlas file (file_atlas). Each value corresponds to a given tract, e.g., corticospinal left.
% NB: 238=WM, 255=CSF (added by jcohen on 2014-12-08)
% label_values = [14 26 38 47 52 62 70 82 89 94 101 107 112 116 121 146 152 159 167 173 180 187 194 199 204 208 214 219 224 230 238 255];
label_left = [14 26 38 47 52 62 70 82 89 94 101 107 112 116 121 45 80 120];
label_right = [146 152 159 167 173 180 187 194 199 204 208 214 219 224 230 150 190 220];
label_pve = [255];
label_values = [label_left, label_right, label_pve];

%--------------------------------------------------------------------------
% TEMPLATE PARAMETERS
%--------------------------------------------------------------------------
which_template = 'PAM50'; % 'MNI-Poly-AMU'
if strcmp(which_template, 'MNI-Poly-AMU')
    % MNI-Poly-AMU template:
    % corresponds to mid-C4 in the MNI-Poly-AMU template
    z_slice_ref = 387;
    % interpolation factor for the MNI-Poly-AMU template in order to match the hi-res grays atlas
    interp_factor = 6;
    % these are the value corresponding to the slice number (z) on the MNI-Poly-AMU template, at which the atlas will be warped. It corresponds to the mid-levels as well as the level of the intervertebral disks.
    % NB: to extract these values, you have to look at the T2 and WM template, because this script will crop the WM template (which can be smaller than the T2), therefore the maximum z cannot exceed the zmax that will be generated in the cropped version of the WM template.
    z_disks_mid = [483 476 466 455 440 423 406 387 371 356 339 324 303 286 268 248 229 208 186 166 143 122 98 79 53 35 13 0];
    % same as before-- except that C4 mid-vertebral is not listed. 
    z_disks_mid_noC4 = [483 476 466 455 440 423 406 371 356 339 324 303 286 268 248 229 208 186 166 143 122 98 79 53 35 13 0];
elseif strcmp(which_template, 'PAM50')
    % PAM50 template:
    % corresponds to mid-C4 in the template
    z_slice_ref = 837;
    % interpolation factor for the template in order to match the hi-res grays atlas
    interp_factor = 6;
    crop_size = '43x33x1100vox'; % size to crop the template (only along X and Y) for computational reasons
    % values of the label in the atlas file (file_atlas). Each value corresponds to a given tract, e.g., corticospinal left.
    % NB: 238=WM, 255=CSF (added by jcohen on 2014-12-08)
    % these are the value corresponding to the slice number (z) on the template, at which the atlas will be warped. It corresponds to the levels of the intervertebral disks.
    % NB: to extract these values, you have to look at the T2 and WM template, because this script will crop the WM template (which can be smaller than the T2), therefore the maximum z cannot exceed the zmax that will be generated in the cropped version of the WM template
    z_disks_mid = [151 185 246 301 355 409 460 509 557 601 641 682 721 757 789 823 855 891 921 945];
end

%--------------------------------------------------------------------------
% OUTPUT PARAMETERS
%--------------------------------------------------------------------------
folder_tracts = ['WMtracts_outputs/'];
folder_ctrl = ['registered_template/'];
folder_final = ['final_results/'];
prefix_out = 'PAM50_atlas';
ext = '.nii.gz'; % .nii.gz. N.B. THIS NEEDS TO BE IN nii.gz BECAUSE THE FUNCTION save_avw SAVES IN nii.gz!!!
fsloutputype = 'export FSLOUTPUTTYPE=NIFTI_GZ; ';


%--------------------------------------------------------------------------
%----------------------- Starting the pipeline -------------------------
%--------------------------------------------------------------------------

template_mask = [file_template '_mask'];
template_cropped = [file_template '_c6v'];
template_cropped_interp = [template_cropped '_int' num2str(interp_factor)];
template_cig = [template_cropped '_intgauss' num2str(interp_factor)];
templateci_slice_ref = [template_cropped_interp '_slice_ref'];
templateci_slice_ref_thresh = [templateci_slice_ref '_thresh'];
templateci_sr_nohd = [templateci_slice_ref '_nohd'];
templateci_thresh = [template_cropped_interp '_thrp6'];
templateci_srt_nohd = [templateci_slice_ref_thresh '_nohd'];
templatecit_slice_ref = [templateci_thresh '_slice_ref'];

tracts_atlas = cell(1,length(label_values));
mask_nohd = [file_mask];
atlas_nifti = [file_atlas];

num_slice_ref = z_slice_ref + 1;
perc_up = 100*interp_factor;
perc_dn = 100/interp_factor;
prefix_ants = ['reg_'];
prefix_ants_ref = ['reg_ref_'];
affine_atlas = 'Affine0GenericAffine.mat';  %[prefix_ants_ref 'Affine.txt'];
Warp_atlas = [prefix_ants 'Warp_init' ext];
Warp_tmp = [prefix_ants 'Warp_init'];
suffix_ants = '_reg';

% if folder exists, delete it
if exist(path_out, 'dir')
    m = input('Output folder already exists. Delete it? (y/n) ', 's');
    if m == 'y'
        cmd = ['rm -rf ' path_out];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
    end
end
% create output folder
mkdir(path_out);
% go to path output
cd(path_out)
% create subfolders
mkdir(folder_tracts);
mkdir(folder_ctrl);
mkdir(folder_final);


%--------------------------------------------------------------------------
%--- Preliminary operations: cropping and interpolation of the template ---

% go to WMtracts folder
cd(folder_tracts)

% Thresholding the template
cmd = ['c3d ' path_template file_template ext ' -threshold -inf 0.001 0 1 -o ' template_mask ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

cmd = ['c3d ' path_template file_template ext ' ' template_mask ext ' -multiply -o ' template_mask ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Cropping the template
% cmd = ['c3d ' template_mask ext ' -trim 6vox -o ' template_cropped ext];
cmd = ['c3d ' template_mask ext ' -trim-to-size ' crop_size ' -o ' template_cropped ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% Interpolation of the template
cmd = ['c3d ' template_cropped ext ' -interpolation Linear -resample ',...
    num2str(perc_up) 'x' num2str(perc_up) 'x100% -o ' template_cropped_interp ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% Extract reference slice
cmd = ['c3d ' template_cropped_interp ext ' -slice z ' num2str(z_slice_ref) ' -o ' templateci_slice_ref ext];
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
cmd = ['c3d ' templateci_slice_ref ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_slice_ref_thresh ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
% change field dim0 from 3 to 2
cmd = [fsloutputype 'fslroi ' templateci_slice_ref_thresh ' ' templateci_slice_ref_thresh ' 0 -1 0 -1 0 -1'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% Binarization of the template for slice coregistration
cmd = ['c3d ' template_cropped_interp ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_thresh ext];
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
    tract_atlas = [ 'tract_atlas_' num2str(label)];
    save_avw(tracts_atlas{label},tract_atlas,'d',scales);
    % copy header info template --> tract_atlas
    cmd = [fsloutputype 'fslcpgeom ' templateci_slice_ref_thresh ' ' tract_atlas ' -d '];
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
cmd = [fsloutputype 'fslcpgeom ' templateci_slice_ref_thresh ' ' mask_nohd ' -d '];
disp(cmd); [status,result]=unix(cmd); if(status), error(result); end
% copy header info template --> atlas
cmd = [fsloutputype 'fslcpgeom ' templateci_slice_ref_thresh ' ' atlas_nifti ' -d '];
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
cmd = ['isct_antsRegistration --dimensionality 2 -m MeanSquares[' templateci_slice_ref_thresh ext ',' mask_nohd ext ',1,4] -t Affine[1] --convergence 100x10 -s 0x0 -f 2x1 -r [' templateci_slice_ref_thresh ext ',' mask_nohd ext ', 0] -o [Affine,' mask_nohd '_affine' ext ']'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)

% estimate diffeomorphic transformation
cmd =['isct_antsRegistration --dimensionality 2 --initial-moving-transform '  affine_atlas ' ',...
    '--transform SyN[0.1,3,0] --metric MeanSquares[' templateci_slice_ref_thresh ext ',' mask_nohd ext ',1,4] ',... 
    '--convergence 100x10 --shrink-factors 4x1 --smoothing-sigmas 0x0mm ',...
    '--output [' prefix_ants_ref ',' mask_nohd '_affine_warp' ext '] --collapse-output-transforms 1'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)

% Rename warping field
movefile([prefix_ants_ref '1Warp.nii.gz'],[Warp_tmp ext]);

% Constraint the warping field to preserve symmetry
cmd = ['c3d -mcs ' Warp_tmp ext ' -oo ' Warp_tmp 'x' ext ' ' Warp_tmp 'y' ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

[warpx,dims,scales] = read_avw([Warp_tmp 'x' ext]);
warpy = read_avw([Warp_tmp 'y' ext]);
warpx = (warpx - warpx(end:-1:1,:)) / 2;
warpy = (warpy + warpy(end:-1:1,:)) / 2;

save_avw(warpx,[Warp_tmp 'xa' ext],'d',scales);
save_avw(warpy,[Warp_tmp 'ys' ext],'d',scales);

cmd = ['c3d ' Warp_tmp 'x' ext ' ' Warp_tmp 'xa' ext ' -copy-transform -o ' Warp_tmp 'xa' ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

cmd = ['c3d ' Warp_tmp 'y' ext ' ' Warp_tmp 'ys' ext ' -copy-transform -o ' Warp_tmp 'ys' ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

% Applying tranform to the mask
cmd = ['isct_antsApplyTransforms -d 2 -i ' mask_nohd ext ' -o ' mask_nohd suffix_ants ext ' -t ' Warp_atlas ' '  affine_atlas ' -r ' templateci_slice_ref_thresh ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

% Applying tranform to the initial atlas
cmd = ['isct_antsApplyTransforms -d 2 -i ' atlas_nifti ext ' -o ' atlas_nifti suffix_ants ext ' -t ' Warp_atlas ' '  affine_atlas ' -r ' templateci_slice_ref_thresh ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

% Applying tranform to the tract files and copying geometry and saving
for label = 1:length(label_left)
    label_l = label;
    label_r = label+length(label_left);
    tract_atlas_g = [ 'tract_atlas_' num2str(label_l)];
    tract_atlas_d = [ 'tract_atlas_' num2str(label_r)];
    
    cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_g ext ' -o ' tract_atlas_g suffix_ants ext ' -t ' Warp_atlas ' '  affine_atlas ' -r ' templateci_slice_ref_thresh ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    
    cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_d ext ' -o ' tract_atlas_d suffix_ants ext ' -t ' Warp_atlas ' '  affine_atlas ' -r ' templateci_slice_ref_thresh ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    
    tract_reg_g = [ 'tract_atlas_' num2str(label_l) suffix_ants];
    temp_g = read_avw(tract_reg_g);
    
    tract_reg_d = [ 'tract_atlas_' num2str(label_r) suffix_ants];
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
    tract_atlas = [ 'tract_atlas_' num2str(label)];
    
    cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas ext ' -o ' tract_atlas suffix_ants ext ' -t ' Warp_atlas ' '  affine_atlas ' -r ' templateci_slice_ref_thresh ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    
    tract_reg_g = [ 'tract_atlas_' num2str(label) suffix_ants];
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



cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(z_slice_ref) ' -o ' templatecit_slice_ref ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

[mstatus,msg] = copyfile([templateci_slice_ref ext],[template_cropped_interp '_slice' z_slice_ref ext]);
if(~mstatus), error(msg); end

reference_slice = [template_cropped_interp '_slice' z_slice_ref];
%thr_ref_slice = templatecit_slice_ref;

% insert z_ref into vector list
izref = min(find(z_slice_ref < z_disks_mid));
z_disks_mid = [z_disks_mid(1:izref-1), z_slice_ref, z_disks_mid(izref:end)];

nb_slices = length(z_disks_mid);

fprintf('\n*** Register slice i+1 to slice i ***\n')
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
    cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(zslice) ' -o ' templatecit_slice ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

    % extract slice corresponding to z=zslice+1 (for neighrest-slice registration)
    cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(zslicenext) ' -o ' templatecit_slicenext ext];
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
        '--transform BSplineSyN[0.2,3] --metric MeanSquares[' templatecit_slicenext ext ',' templatecit_slice ext ',1,4] ',... 
        '--convergence 100x5 --shrink-factors 2x1 --smoothing-sigmas 5x5vox ',...
        '--output [' [prefix_ants num2str(zslice) 'to' num2str(zslicenext) '_'] ',' prefix_ants 'slicenext_to_slice.nii.gz]'];
%     cmd =['isct_antsRegistration --dimensionality 2 ',...
%         '--transform BSplineSyN[0.2,3] --metric MeanSquares[' templatecit_slice ext ',' templatecit_slicenext ext ',1,4] ',... 
%         '--convergence 100x20 --shrink-factors 2x1 --smoothing-sigmas 0x0vox ',...
%         '--output [' [prefix_ants num2str(zslice) '_'] ',' prefix_ants 'slicenext_to_slice.nii.gz]'];    
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
end

fprintf('\n*** Concatenate warping fields for each slice ***\n')
% at this point
% reg_476to483_0Warp.nii.gz is the warping field for 476 --> 483
% reg_476to483_0InverseWarp.nii.gz is the warping field for 476 <-- 483
% the goal now, is to concatenate warping fields, to obtain "reg_concat_refto483": zref --> 483 
% and so on...
% so the concatenation should be done like that:
% if zslice > zref (ex: 950 > 830)
%   reg_concat_refto950 = reg_refto870_0Warp + ... + reg_920to950_0Warp
% if zslice < zref (ex: 300 < 830)
%   reg_concat_refto300 = reg_800toref_0InverseWarp + reg_750to800_0InverseWarp + ... + reg_300to350_0InverseWarp
nb_slices = length(z_disks_mid);
for iz = 1:nb_slices
    
%     templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
%     zslice = z_disks_mid_noC4(iz);
    zslice = z_disks_mid(iz);
    templatecit_slice = [templateci_thresh '_slice' num2str(z_slice_ref)];

    if zslice ~= z_slice_ref
        % output concatenated field is: reg_concat_"zslice"
        cmd = ['isct_ComposeMultiTransform 2 ', prefix_ants, 'concat_refto', num2str(zslice), ext, ' -R ', templatecit_slice, ext];

        if zslice > z_slice_ref
            % if zslice is superior to ref slice, then concatenate forward warping
            % fields, from the zref to zslice.
            % concatenated warping field: warp_temp = reg_0Warp
            for izconcat = izref:iz-1
                cmd = [cmd, ' ', prefix_ants num2str(z_disks_mid(izconcat)) 'to' num2str(z_disks_mid(izconcat+1)), '_0Warp', ext];
            end
        elseif zslice < z_slice_ref
            % if zslice is inferior to ref slice, then concatenate backward warping
            % fields, from the zref to zslice.
            for izconcat = izref-1:-1:iz
                cmd = [cmd, ' ', prefix_ants num2str(z_disks_mid(izconcat)) 'to' num2str(z_disks_mid(izconcat+1)), '_0InverseWarp', ext];
            end
        end
        % run command
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
    else
        % zslice = z_slice_ref: do nothing
    end
end


fprintf('\n*** Adjust warping fields to minimize the propagation of error from concatenation ***\n')
% nb_slices = length(z_disks_mid_noC4);
for iz = 1:nb_slices

    zslice = z_disks_mid(iz);
    templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    warp_slice = [ prefix_ants, 'concat_refto', num2str(zslice), ext];

    if zslice ~= z_slice_ref
% 
%     zslice = z_disks_mid_noC4(iz);    
% 
%     templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
%     warp_slice = [ prefix_ants, 'concat_', num2str(zslice), ext];
%     izref = find(z_disks_mid == z_slice_ref);
    
        % register refslice to zslice using concatenated warping field as
        % initial transformation
        cmd =['isct_antsRegistration --dimensionality 2 ',...
            '--initial-moving-transform ', warp_slice, ' ',...
            '--transform BSplineSyN[0.2,3] --metric MeanSquares[' templatecit_slice ext ',' templatecit_slice_ref ext ',1,4] ',... 
            '--convergence 200x5 --shrink-factors 2x1 --smoothing-sigmas 5x5vox ',...
            '--output [' prefix_ants, 'concat_refto', num2str(zslice) '_,' templatecit_slice_ref 'to' num2str(zslice) ext ']' ];
        disp(cmd)
        [status,result] = unix(cmd);
    %     disp(result)
        if(status),error(result);end 

        % Replace the concatenated warping field with the new warping field 
         movefile([prefix_ants, 'concat_refto', num2str(zslice), '_0Warp', ext], warp_slice)
    else
        % zslice = z_slice_ref: do nothing
    end        
end


disp('*** Symmetrize warping fields ***')
% nb_slices = length(z_disks_mid_noC4);
for iz = 1:nb_slices
    zslice = z_disks_mid(iz);
    if zslice ~= z_slice_ref
    
        warp_temp = [ prefix_ants, 'concat_refto', num2str(zslice)];
        warp_slice = [ prefix_ants, 'concat_sym_refto', num2str(zslice), ext];

        % Constraint the warping field to preserve symmetry
        cmd = ['c3d -mcs ' warp_temp ext ' -oo ' warp_temp 'x' ext ' ' warp_temp 'y' ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)

        % read warping field
        [warpx,dims,scales] = read_avw([warp_temp 'x' ext]);
        warpy = read_avw([warp_temp 'y' ext]);
        warpx = (warpx - warpx(end:-1:1,:)) / 2;
        warpy = (warpy + warpy(end:-1:1,:)) / 2;

        save_avw(warpx,[warp_temp 'xa' ext],'d',scales);
        save_avw(warpy,[warp_temp 'ys' ext],'d',scales);

        cmd = ['c3d ' warp_temp 'x' ext ' ' warp_temp 'xa' ext ' -copy-transform -o ' warp_temp 'xa' ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)

        cmd = ['c3d ' warp_temp 'y' ext ' ' warp_temp 'ys' ext ' -copy-transform -o ' warp_temp 'ys' ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)

        cmd = ['c3d ' warp_temp 'xa' ext ' ' warp_temp 'ys' ext ' -omc 2 ' warp_slice];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    end
end


disp('*** Applies warping fields for registration to each intermediate slice ***')
% nb_slices = length(z_disks_mid_noC4)
for iz = 1:nb_slices
    
    disp(['SLICE #: ', num2str(iz), '/', num2str(nb_slices)])
    
    zslice = z_disks_mid(iz);

    if zslice ~= z_slice_ref

        numSlice = zslice+1;
        templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
        atlas_slice = [atlas_nifti suffix_ants num2str(zslice)];
        warp_slice = [ prefix_ants, 'concat_sym_refto', num2str(zslice), ext];

        % Apply transform to reference slice as a control
        cmd = ['isct_antsApplyTransforms -d 2 -i ' templatecit_slice_ref ext ' -o ' templatecit_slice_ref suffix_ants num2str(zslice) ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
        % run command and check results. Note: here we cannot use status,
        % because if isct_antsApplyTransforms errors, then status=0.
        disp(cmd); [status,result] = unix(cmd); if(~exist([templatecit_slice_ref suffix_ants num2str(zslice) ext])), error(result); end

        % Apply transform to the initial atlas as a control
    %     cmd = ['sct_WarpImageMultiTransform 2 ' atlas_nifti ext ' ' atlas_slice ext ' ' warp_slice ' ' Warp_atlas ' '  affine_atlas ' -R ' templatecit_slice ext];
        cmd = ['isct_antsApplyTransforms -d 2 -i ' atlas_nifti ext ' -o ' atlas_slice ext ' -t ' Warp_atlas ' '  affine_atlas ' -r ' templateci_slice_ref_thresh ext];
        disp(cmd); [status,result] = unix(cmd); if(~exist([atlas_slice ext])), error(result); end
        cmd = ['c3d ' templatecit_slice_ref ' ' atlas_slice ext ' -copy-transform -o ' atlas_slice ext];  % copy geom-- added: 2014-08-30
        disp(cmd); [status,result] = unix(cmd); if(~exist([atlas_slice ext])), error(result); end
        cmd = ['isct_antsApplyTransforms -d 2 -i ' atlas_slice ext ' -o ' atlas_slice suffix_ants ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
        disp(cmd); [status,result] = unix(cmd); if(~exist([atlas_slice suffix_ants ext])), error(result); end

        % Apply tranform to the WM tract files and constraint to be symmetric
        for label = 1:length(label_left)
            label_l = label;
            label_r = label+length(label_left);
            tract_atlas_g = [ 'tract_atlas_' num2str(label_l)];
            tract_atlas_d = [ 'tract_atlas_' num2str(label_r)];

            % LEFT
            cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_g ext ' -o ' tract_atlas_g suffix_ants ext ' -t ' Warp_atlas ' '  affine_atlas ' -r ' templateci_slice_ref_thresh ext];
            disp(cmd); [status,result] = unix(cmd); if(~exist([tract_atlas_g suffix_ants ext])), error(result); end
            cmd = ['c3d ' templatecit_slice_ref ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext ext];  % copy geom-- added: 2014-08-30
            disp(cmd); [status,result]=unix(cmd); if(status), error(result); end, %disp(result)
            cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_g suffix_ants ext ' -o ' tract_atlas_g suffix_ants ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
            disp(cmd); [status,result] = unix(cmd); if(~exist([tract_atlas_g suffix_ants ext])), error(result); end

            % RIGHT
            cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_d ext ' -o ' tract_atlas_d suffix_ants ext ' -t ' Warp_atlas ' '  affine_atlas ' -r ' templateci_slice_ref_thresh ext];
            disp(cmd); [status,result] = unix(cmd); if(~exist([tract_atlas_d suffix_ants ext])), error(result); end
            cmd = ['c3d ' templatecit_slice_ref ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext ext];  % copy geom-- added: 2014-08-30
            disp(cmd); [status,result]=unix(cmd); if(status), error(result); end, %disp(result)
            cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas_d suffix_ants ext ' -o ' tract_atlas_d suffix_ants ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
            disp(cmd); [status,result] = unix(cmd); if(~exist([tract_atlas_d suffix_ants ext])), error(result); end

            % copy header from template to registered atlas
            % NB: changed templateci_slice to templatecit_slice (2014-08-04)
            cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
            disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

            cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
            disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

            tract_reg_g = [ 'tract_atlas_' num2str(label_l) suffix_ants];
            temp_g = read_avw(tract_reg_g);

            tract_reg_d = [ 'tract_atlas_' num2str(label_r) suffix_ants];
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
            tract_atlas = [ 'tract_atlas_' num2str(label)];

            cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas ext ' -o ' tract_atlas suffix_ants ext ' -t ' Warp_atlas ' '  affine_atlas ' -r ' templateci_slice_ref_thresh ext];
            disp(cmd); [status,result] = unix(cmd); if(~exist([tract_atlas suffix_ants ext])), error(result); end
            cmd = ['c3d ' templatecit_slice_ref ' ' tract_atlas suffix_ants ext ' -copy-transform -o ' tract_atlas suffix_ants ext ext];  % copy geom-- added: 2014-08-30
            disp(cmd); [status,result]=unix(cmd); if(status), error(result); end, %disp(result)
            cmd = ['isct_antsApplyTransforms -d 2 -i ' tract_atlas suffix_ants ext ' -o ' tract_atlas suffix_ants ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
            disp(cmd); [status,result] = unix(cmd); if(~exist([tract_atlas suffix_ants ext])), error(result); end

            % copy header from template to registered atlas
            % NB: changed templateci_slice to templatecit_slice (2014-08-04)
            cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas suffix_ants ext ' -copy-transform -o ' tract_atlas suffix_ants ext];
            disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)

            tract_reg = [ 'tract_atlas_' num2str(label) suffix_ants];
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
        movefile([reg_slice_current ext], [path_out folder_ctrl]);
        movefile([atlas_slice suffix_ants ext], [path_out folder_ctrl]);
    else
        disp(['Reference slice: skipping this step.'])
    end   
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

%% open tract description file
disp('*** Open label description file ***')
fid = fopen([path_atlas_data, file_atlas_txt]);
tline = fgetl(fid);
struct_label = {};
i=1;
while ischar(tline)
    disp(tline)
    tline_split = strsplit(tline,',');
    struct_label{i}.id = tline_split{1};
    struct_label{i}.value = tline_split{2};
    struct_label{i}.description = tline_split{3};
    % go to next line
    tline = fgetl(fid);
    i = i+1;
end
fclose (fid);

%% Loop on labels to compute partial volume values without HR version
disp('*** Compute partial volume and save data ***')
% loop across tracts
for label = 1:length(label_values)
    % Compute label file name
    for i = 1:length(struct_label)
        if str2num(struct_label{i}.value) == label_values(label)
            % Build suffix
            numlabel = sprintf('%02d',str2num(struct_label{i}.id));
            disp(['Label value: ',num2str(label_values(label)),', Label ID: ', numlabel,', Description: ',struct_label{i}.description])
            % build file name
            filetractML = [path_out,folder_final,prefix_out,'_',numlabel];
%             struct_label{label}.filename = [prefix_out,'_',numlabel,ext];
            break
        end
    end
    % Save ML version and copy geometry
    save_avw(tracts{label},filetractML,'d',scalesCROP);
    cmd = ['c3d ' template_cropped ext ' ' filetractML ext ' -copy-transform -o ' filetractML ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
	% Reslice into native template space
	cmd = ['c3d ' path_template file_template ext ' ' filetractML ext ' -reslice-identity -o ' filetractML ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(resul)
    % copy geometry from white matter template
    cmd = [fsloutputype 'fslcpgeom ', path_template, file_template, ' ', filetractML];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
end

%% Create global WM and GM atlases
% Build global WM atlas
description = {'WM'};
global_suffix = '_wm';
tract_global = zeros(size(tracts{1}));
description = {'WM', 'GM'};
global_suffix = '_cord';
tract_global = zeros(size(tracts{1}));
for ilabel = 1:length(struct_label)
    for idescription = 1:length(description)
        if findstr(struct_label{ilabel}.description,description{idescription})
            % loop across label_values to make sure to get the right value
            for jlabel = 1:length(label_values)
                if label_values(jlabel) == str2num(struct_label{ilabel}.value)
                    disp(['Adding: ',struct_label{ilabel}.description,' (value=',struct_label{ilabel}.value,')'])
                    tract_global = tract_global + tracts{jlabel};
                end
            end
        end
    end
end
% Save image and copy geometry
filetractML = [path_out,folder_final,which_template,global_suffix];
save_avw(tract_global,filetractML,'d',scalesCROP);
cmd = ['c3d ' template_cropped ext ' ' filetractML ext ' -copy-transform -o ' filetractML ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
% Reslice into native template space
cmd = ['c3d ' path_template file_template ext ' ' filetractML ext ' -reslice-identity -o ' filetractML ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(resul)
% copy geometry from white matter template
cmd = [fsloutputype 'fslcpgeom ', path_template, file_template, ' ', filetractML];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% Build global GM atlas
description = {'GM'};
global_suffix = '_gm';
tract_global = zeros(size(tracts{1}));
description = {'WM', 'GM'};
global_suffix = '_cord';
tract_global = zeros(size(tracts{1}));
for ilabel = 1:length(struct_label)
    for idescription = 1:length(description)
        if findstr(struct_label{ilabel}.description,description{idescription})
            % loop across label_values to make sure to get the right value
            for jlabel = 1:length(label_values)
                if label_values(jlabel) == str2num(struct_label{ilabel}.value)
                    disp(['Adding: ',struct_label{ilabel}.description,' (value=',struct_label{ilabel}.value,')'])
                    tract_global = tract_global + tracts{jlabel};
                end
            end
        end
    end
end
% Save image and copy geometry
filetractML = [path_out,folder_final,which_template,global_suffix];
save_avw(tract_global,filetractML,'d',scalesCROP);
cmd = ['c3d ' template_cropped ext ' ' filetractML ext ' -copy-transform -o ' filetractML ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
% Reslice into native template space
cmd = ['c3d ' path_template file_template ext ' ' filetractML ext ' -reslice-identity -o ' filetractML ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(resul)
% copy geometry from white matter template
cmd = [fsloutputype 'fslcpgeom ', path_template, file_template, ' ', filetractML];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% Build global cord atlas
description = {'WM', 'GM'};
global_suffix = '_cord';
tract_global = zeros(size(tracts{1}));
for ilabel = 1:length(struct_label)
    for idescription = 1:length(description)
        if findstr(struct_label{ilabel}.description,description{idescription})
            % loop across label_values to make sure to get the right value
            for jlabel = 1:length(label_values)
                if label_values(jlabel) == str2num(struct_label{ilabel}.value)
                    disp(['Adding: ',struct_label{ilabel}.description,' (value=',struct_label{ilabel}.value,')'])
                    tract_global = tract_global + tracts{jlabel};
                end
            end
        end
    end
end
% binarize
tract_global(tract_global<0.5) = 0;
tract_global(tract_global>=0.5) = 1;
% Save image and copy geometry
filetractML = [path_out,folder_final,which_template,global_suffix];
save_avw(tract_global,filetractML,'d',scalesCROP);
cmd = ['c3d ' template_cropped ext ' ' filetractML ext ' -copy-transform -o ' filetractML ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
% Reslice into native template space
cmd = ['c3d ' path_template file_template ext ' ' filetractML ext ' -reslice-identity -o ' filetractML ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(resul)
% copy geometry from white matter template
cmd = [fsloutputype 'fslcpgeom ', path_template, file_template, ' ', filetractML];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

%% Create file info_label.txt
disp('*** Create file info_label.txt ***')
% open file
file_infolabel = [path_out,folder_final,'info_label.txt'];
fid = fopen(file_infolabel,'w');
fprintf(fid,'# White matter atlas. Generated on: %s\n',date);
fprintf(fid,'# ID, name, file\n');
% loop across tracts
for label = 1:length(struct_label)
    numlabel = sprintf('%02d',str2num(struct_label{i}.id));
    fprintf(fid,'%s, %s, %s\n',struct_label{label}.id, struct_label{label}.description, [prefix_out,'_',numlabel,ext]);
end
% add combined labels
fprintf(fid,'\n# Combined labels\n');
fprintf(fid,'# ID, name, file\n');
fprintf(fid,'50, spinal cord, 0:35\n');
fprintf(fid,'51, white matter, 0:29\n');
fprintf(fid,'52, gray matter, 30:35\n');
fprintf(fid,'52, dorsal columns, 0:3\n');
fprintf(fid,'53, lateral funiculi, 4:13\n');
fprintf(fid,'54, ventral funiculi, 14:29\n');
fclose(fid);

%% Create global WM and GM atlases
cmd = ['mkdir ',path_out,folder_final,'temp_wm'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
cd([path_out,folder_final,'temp_wm']);
% copy all WM files atlas
cmd = ['cp ../*.nii.gz .'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
% remove first one
cmd = ['rm ',which_template,'_atlas_00.nii.gz'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
% sum all WM tracts
cmd = ['sct_maths -i ../PAM50_atlas_00.nii.gz -add *.nii.gz -o ',which_template,'_wm.nii.gz'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
% copy file and delete temp folder
cmd = ['cp ',which_template,'_wm.nii.gz ../'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
cd('../')
cmd = ['rm -rf temp_wm'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end


%% FINISHED!
bricon = ' -b 0.5,1 '
disp 'Done! To see results, type:'
disp(['cd ',path_out folder_final])
disp(['fslview ',path_template,which_template,'_t2.nii.gz -b 0,4000 ',path_template,which_template,'_wm.nii.gz',bricon,' -l Blue-Lightblue ',prefix_out,'_00.nii.gz -l Red',bricon,prefix_out,'_01.nii.gz -l Green',bricon,prefix_out,'_02.nii.gz -l Blue',bricon,prefix_out,'_03.nii.gz -l Yellow',bricon,prefix_out,'_04.nii.gz -l Pink ',bricon,prefix_out,'_30.nii.gz -l Red-Yellow ',bricon,prefix_out,'_31.nii.gz -l Copper ',bricon,' &'])
