function create_atlas(file_param)
% White matter tracts template construction
% 
% This script creates an atlas of white matter tracts registered to a
% given 3D MRI template of the spinal cord, using a 2D existing atlas.
%
% Usage:
% - Using 'pathtool' add the folder containing this script.
% - Create a folder raw_data/ that contains:
%   - A 2D atlas in single-channel PNG format, with a single value assigned per label.
%     Example: $SCTDIR/dev/atlas/raw_data/atlas_grays_cerv_sym_correc_r6.png
%   - A binarized version of the 2D atlas (use the script binarize_atlas.sh)
%     Example: $SCTDIR/dev/atlas/raw_data/mask_grays_cerv_sym_correc_r5
%   - A CSV text file with the following information: LabelID,LabelValue,Description
%     Note: it is important that 'Description' includes the term 'left' or 'right' for the symmetrization procedure. For the CSF or background, indicate the label value.
%     Example: $SCTDIR/dev/atlas/raw_data/atlas_grays_cerv_sym_correc_r6_label.txt
% - Create a file parameters_NAMEOFTEMPLATE.m
%   Example: parameters_PAM50.m
% - Start MATLAB from the Terminal so that the environment variables for FSL, c3d and ANTs will be declared inside Matlab's environment.
% - run this function:
%   create_atlas(my_param_file.m)
%
% Dependencies:
%   Matlab:         image processing toolbox
%   m_normalize.m   normalization function
%   dnsamplelin.m   function for downsampling by computing mean value for each region
%   m_linear_interp.m
%   FSL
%   c3d
%   ANTs

% load parameters (modify the line below to use your parameters)
run(file_param)  % default: parameters_PAM50.m

% use debugger
dbstop if error
% get path of FSL
[status, path_fsl] = unix('echo $FSLDIR');
% get FSL matlab functions
path_fsl_matlab = strcat(path_fsl, '/etc/matlab');
% add to path
addpath(path_fsl_matlab);
% extension of atlas files. N.B. THIS NEEDS TO BE IN nii.gz BECAUSE THE FUNCTION save_avw SAVES IN nii.gz!
ext = '.nii.gz'; 
% set FSL output to nii.gz
fsloutputype = 'export FSLOUTPUTTYPE=NIFTI_GZ; ';


%--------------------------------------------------------------------------
%----------------------- Starting the pipeline -------------------------
%--------------------------------------------------------------------------

template_mask = [file_template '_mask'];
template_cropped = [file_template '_c6v'];
template_cropped_interp = [template_cropped '_int' num2str(interp_factor)];
template_cig = [template_cropped '_intgauss' num2str(interp_factor)];
templateci_slice_ref = [template_cropped_interp, '_slice', num2str(z_slice_ref)];
templateci_slice_ref_thresh = [templateci_slice_ref '_thresh'];
templateci_sr_nohd = [templateci_slice_ref '_nohd'];
templateci_thresh = [template_cropped_interp '_thrp6'];
templateci_srt_nohd = [templateci_slice_ref_thresh '_nohd'];
templatecit_slice_ref = [templateci_thresh, '_slice', num2str(z_slice_ref)];

mask_nohd = [file_mask];
atlas_nifti = [file_atlas];

num_slice_ref = z_slice_ref + 1;
perc_up = 100*interp_factor;
perc_dn = 100/interp_factor;
prefix_ants = ['warp_'];
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


%% Read tract description file
fprintf('\n*** Open label description file ***\n')
fid = fopen([path_atlas_data, file_atlas_txt]);
tline = fgetl(fid);
struct_label = {};
label_right = [];
label_left = [];
label_pve = [];
i=1;
while ischar(tline)
    disp(tline)
    tline_split = strsplit(tline,',');
    struct_label{i}.id = tline_split{1};
    struct_label{i}.value = tline_split{2};
    struct_label{i}.description = tline_split{3};
    % check if this label is right or left
    if ~isempty(strfind(struct_label{i}.description, 'right'))
        label_right = [label_right, str2num(struct_label{i}.value)];
    elseif ~isempty(strfind(struct_label{i}.description, 'left'))
        label_left = [label_left, str2num(struct_label{i}.value)];
    else
        label_pve = [label_pve, str2num(struct_label{i}.value)];
    end
    % go to next line
    tline = fgetl(fid);
    i = i+1;
end
fclose (fid);
label_values = [label_left, label_right, label_pve];


%% Preliminary operations: cropping and interpolation of the template
fprintf('\n*** Preliminary operations: cropping and interpolation of the template ***\n')

% go to WMtracts folder
cd(folder_tracts)

% Thresholding the template
cmd = ['c3d ' path_template file_template ext ' -threshold -inf 0.001 0 1 -o ' template_mask ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

cmd = ['c3d ' path_template file_template ext ' ' template_mask ext ' -multiply -o ' template_mask ext];
disp(cmd); [status,result] = unix(cmd);
if(status), error(result); end

% Cropping the template
cmd = ['c3d ' template_mask ext ' -trim-to-size ' crop_size ' -o ' template_cropped ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% Interpolation of the template
cmd = ['c3d ' template_cropped ext ' -interpolation Linear -resample ' num2str(perc_up) 'x' num2str(perc_up) 'x100% -o ' template_cropped_interp ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

% Extract reference slice
cmd = ['c3d ' template_cropped_interp ext ' -slice z ' num2str(z_slice_ref) ' -o ' templateci_slice_ref ext];
% cmd = [fsloutputype 'fslroi ' template_cropped_interp ' ' templateci_slice_ref ' 0 -1 0 -1 ' num2str(z_slice_ref) ' 1 '];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
% change field dim0 from 3 to 2
cmd = [fsloutputype 'fslroi ' templateci_slice_ref ' ' templateci_slice_ref ' 0 -1 0 -1 0 -1'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end

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

% Save the atlas and mask into a nifti with the same scales as the template
[slice_ref,~,scales] = read_avw(templateci_slice_ref_thresh);
% [slice_ref,~,scales] = read_avw(templateci_srt_nohd);
atlas = imread([path_atlas_data file_atlas ext_atlas]);
tracts_atlas = cell(1,length(label_values));
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


%% Initial registration of the atlas to the reference slice of the template
fprintf('\n*** Initial registration of the atlas to the reference slice of the template ***\n')

% write initialization affine transfo (empirically found)
fid=fopen('affine_init.txt','w');
fprintf(fid, '#Insight Transform File V1.0\n');
fprintf(fid, '#Transform 0\n');
fprintf(fid, 'Transform: AffineTransform_double_2_2\n');
fprintf(fid, 'Parameters: 2 0 0 2 10 -7\n');
fprintf(fid, 'FixedParameters: 0 -1.5\n');
fclose(fid);

% estimate affine transformation from atlas to template
% here, using flag -r 0 to initialize transformations based on geometrical center of images
cmd = ['isct_antsRegistration --dimensionality 2 -m MeanSquares[' templateci_slice_ref_thresh ext ',' mask_nohd ext ',1,4] -t Affine[1] --convergence 100x100 -s 1x1 -f 2x1 -r affine_init.txt -o [Affine,' mask_nohd '_affine' ext ']'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)

% estimate diffeomorphic transformation
cmd =['isct_antsRegistration --dimensionality 2 --initial-moving-transform '  affine_atlas ' ',...
    '--transform SyN[0.1,3,0] --metric MeanSquares[' templateci_slice_ref_thresh ext ',' mask_nohd ext ',1,4] ',... 
    '--convergence 100x10 --shrink-factors 4x1 --smoothing-sigmas 0x0mm ',...
    '--output [' prefix_ants ',' mask_nohd '_affine_warp' ext '] --collapse-output-transforms 1'];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)

% Rename warping field
movefile([prefix_ants '1Warp.nii.gz'],[Warp_tmp ext]);

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


%% Construction of the template
fprintf('\n*** Construction of the template ***\n')
% extract reference slice
cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(z_slice_ref) ' -o ' templatecit_slice_ref ext];
disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
[mstatus,msg] = copyfile([templateci_slice_ref ext],[template_cropped_interp '_slice' z_slice_ref ext]);
if(~mstatus), error(msg); end
reference_slice = [template_cropped_interp '_slice' z_slice_ref];

% insert z_ref into vector list
izref = min(find(z_slice_ref < z_disks_mid));
z_disks_mid = [z_disks_mid(1:izref-1), z_slice_ref, z_disks_mid(izref:end)];
nb_slices = length(z_disks_mid);

% Move up: ref --> max(z_disks_mid)
fprintf('\n*** Register slice i+1 to slice i ***\n')
for iz = izref:nb_slices-1
    % identify current and next slice
    zslice = z_disks_mid(iz);
    zslicenext = z_disks_mid(iz+1);
    % build file names
    templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    templatecit_slicenext = [templateci_thresh '_slice' num2str(zslicenext)];
    % extract slice corresponding to z=zslice+1
    cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(zslicenext) ' -o ' templatecit_slicenext ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
    % apply previous transfo to ref (except if iz == izref)
    if iz == izref
        file_moving = [templatecit_slice_ref ext];
        warp_ref2slice = '';
    else
        warp_ref2slice = [prefix_ants, num2str(z_slice_ref), 'to', num2str(zslice), '.nii.gz'];
        cmd = ['isct_antsApplyTransforms -d 2 -i ', templatecit_slice_ref, ext, ' -o ', templatecit_slice_ref, 'to', num2str(zslice), ext, ' -t ', warp_ref2slice, ' -r ', templatecit_slice, ext];
        % run command and check results. Note: here we cannot use status because if isct_antsApplyTransforms errors, then status=0.
        disp(cmd); [status, result] = unix(cmd); if(~exist([templatecit_slice_ref, 'to', num2str(zslice), ext])), error(result); end
        file_moving = [templatecit_slice_ref 'to' num2str(zslice) ext];
    end
    % estimate transformation slice->slicenext
    cmd =['isct_antsRegistration --dimensionality 2 ',...
        '--transform Affine[0.5] --metric MeanSquares[' templatecit_slicenext ext ',' file_moving ',1,4] ',... 
        '--convergence 100x100 --shrink-factors 2x1 --smoothing-sigmas 1x0vox ',...
        '--transform BSplineSyN[0.2,3] --metric MeanSquares[' templatecit_slicenext ext ',' file_moving ',1,4] ',... 
        '--convergence 500x10 --shrink-factors 2x1 --smoothing-sigmas 1x0vox ',...
        '--output [' prefix_ants num2str(z_slice_ref) 'to' num2str(zslicenext) '_,' templatecit_slice_ref 'to' num2str(zslicenext) ext ']'];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
    % concat ref->slice and slice->slicenext
    cmd = ['isct_ComposeMultiTransform 2 ', prefix_ants, num2str(z_slice_ref), 'to', num2str(zslicenext), ext, ' -R ', templatecit_slice, ext, ' ', prefix_ants, num2str(z_slice_ref), 'to', num2str(zslicenext), '_0GenericAffine.mat ', prefix_ants, num2str(z_slice_ref), 'to', num2str(zslicenext), '_1Warp.nii.gz ', warp_ref2slice];
    disp(cmd); [status,result] = unix(cmd);
end

% Move down: ref --> min(z_disks_mid)
fprintf('\n*** Register slice i-1 to slice i ***\n')
for iz = izref:-1:2
    % identify current and next slice
    zslice = z_disks_mid(iz);
    zslicenext = z_disks_mid(iz-1);
    % build file names
    templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    templatecit_slicenext = [templateci_thresh '_slice' num2str(zslicenext)];
    % extract slice corresponding to z=zslice+1
    cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(zslicenext) ' -o ' templatecit_slicenext ext];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, %disp(result)
    % apply previous transfo to ref (except if iz == izref)
    if iz == izref
        file_moving = [templatecit_slice_ref ext];
        warp_ref2slice = '';
    else
        warp_ref2slice = [prefix_ants, num2str(z_slice_ref), 'to', num2str(zslice), '.nii.gz'];
        cmd = ['isct_antsApplyTransforms -d 2 -i ', templatecit_slice_ref, ext, ' -o ', templatecit_slice_ref, 'to', num2str(zslice), ext, ' -t ', warp_ref2slice, ' -r ', templatecit_slice, ext];
        % run command and check results. Note: here we cannot use status because if isct_antsApplyTransforms errors, then status=0.
        disp(cmd); [status, result] = unix(cmd); if(~exist([templatecit_slice_ref, 'to', num2str(zslice), ext])), error(result); end
        file_moving = [templatecit_slice_ref 'to' num2str(zslice) ext];
    end
    % estimate transformation slice->slicenext
    cmd =['isct_antsRegistration --dimensionality 2 ',...
        '--transform Affine[0.5] --metric MeanSquares[' templatecit_slicenext ext ',' file_moving ',1,4] ',... 
        '--convergence 100x100 --shrink-factors 2x1 --smoothing-sigmas 1x0vox ',...
        '--transform BSplineSyN[0.2,3] --metric MeanSquares[' templatecit_slicenext ext ',' file_moving ',1,4] ',... 
        '--convergence 500x10 --shrink-factors 2x1 --smoothing-sigmas 1x0vox ',...
        '--output [' prefix_ants num2str(z_slice_ref) 'to' num2str(zslicenext) '_,' templatecit_slice_ref 'to' num2str(zslicenext) ext ']'];
    disp(cmd); [status,result] = unix(cmd); if(status), error(result); end
    % concat ref->slice and slice->slicenext
    cmd = ['isct_ComposeMultiTransform 2 ', prefix_ants, num2str(z_slice_ref), 'to', num2str(zslicenext), ext, ' -R ', templatecit_slice, ext, ' ', prefix_ants, num2str(z_slice_ref), 'to', num2str(zslicenext), '_0GenericAffine.mat ', prefix_ants, num2str(z_slice_ref), 'to', num2str(zslicenext), '_1Warp.nii.gz ', warp_ref2slice];
%      cmd = ['isct_ComposeMultiTransform 2 ', prefix_ants, num2str(z_slice_ref), 'to', num2str(zslicenext), ext, ' -R ', templatecit_slice, ext, ' ', prefix_ants num2str(z_slice_ref) 'to' num2str(zslicenext) '_0Warp.nii.gz ', warp_ref2slice];
    disp(cmd); [status,result] = unix(cmd);
end


%% Symmetrize warping fields
fprintf('\n*** Symmetrize warping fields ***\n')
% nb_slices = length(z_disks_mid_noC4);
for iz = 1:nb_slices
    zslice = z_disks_mid(iz);
    if zslice ~= z_slice_ref
        % build file names
        warp_temp = [ prefix_ants, num2str(z_slice_ref), 'to', num2str(zslice)];
        warp_slice = [ prefix_ants, 'sym_', num2str(z_slice_ref), 'to', num2str(zslice)];
        % Constraint the warping field to preserve symmetry
        cmd = ['c3d -mcs ' warp_temp ext ' -oo ' warp_temp 'x' ext ' ' warp_temp 'y' ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
        % read warping field
        [warpx,dims,scales] = read_avw([warp_temp 'x' ext]);
        warpy = read_avw([warp_temp 'y' ext]);
        % do the mean across x and y 
        warpx = (warpx - warpx(end:-1:1,:)) / 2;
        warpy = (warpy + warpy(end:-1:1,:)) / 2;
        % save
        save_avw(warpx,[warp_temp 'xa' ext],'d',scales);
        save_avw(warpy,[warp_temp 'ys' ext],'d',scales);
        % copy transforms for x
        cmd = ['c3d ' warp_temp 'x' ext ' ' warp_temp 'xa' ext ' -copy-transform -o ' warp_temp 'xa' ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
        % copy transforms for y
        cmd = ['c3d ' warp_temp 'y' ext ' ' warp_temp 'ys' ext ' -copy-transform -o ' warp_temp 'ys' ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
        % combine x and y fields as single composite transfo
        cmd = ['c3d ' warp_temp 'xa' ext ' ' warp_temp 'ys' ext ' -omc 2 ' warp_slice ext];
        disp(cmd); [status,result] = unix(cmd); if(status), error(result); end, disp(result)
    end
end


%% Applies warping fields for registration to each intermediate slice
disp('*** Applies warping fields for registration to each intermediate slice ***')
% nb_slices = length(z_disks_mid_noC4)
for iz = 1:nb_slices
    % display stuff
    disp(['SLICE #: ', num2str(iz), '/', num2str(nb_slices)])
    % select slice
    zslice = z_disks_mid(iz);
    % loop across slices
    if zslice ~= z_slice_ref
        numSlice = zslice+1;
        templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
        atlas_slice = [atlas_nifti suffix_ants num2str(zslice)];
        warp_slice = [ prefix_ants, 'sym_', num2str(z_slice_ref), 'to', num2str(zslice), ext];
        % Apply transform to reference slice as a control
        cmd = ['isct_antsApplyTransforms -d 2 -i ' templatecit_slice_ref ext ' -o ' templatecit_slice_ref suffix_ants num2str(zslice) ext ' -t ' warp_slice ' -r ' templatecit_slice ext];
        % run command and check results. Note: here we cannot use status,
        % because if isct_antsApplyTransforms errors, then status=0.
        disp(cmd); [status,result] = unix(cmd); if(~exist([templatecit_slice_ref suffix_ants num2str(zslice) ext])), error(result); end
        % Apply transform to the initial atlas as a control
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

%% Interpolation between computed slices
disp('*** Interpolation between computed slices ***')
for label = 1:length(label_values)
    disp(['LABEL: ', num2str(label), '/', num2str(length(label_values))])
    for k = 1:length(z_disks_mid)-1
        tractsHR{label} = m_linear_interp(tractsHR{label},z_disks_mid(k)+1,z_disks_mid(k+1)+1);
    end
end

%% Downsampling and partial volume computation
disp('*** Downsampling and partial volume computation ***')
max_indx = max(z_disks_mid(:));
for label = 1:length(label_values)
    disp(['LABEL: ', num2str(label), '/', num2str(length(label_values))])
    for zslice = 0:max_indx
        numSlice = zslice+1;
        tracts{label}(:,:,numSlice) = dnsamplelin(tractsHR{label}(:,:,numSlice),interp_factor);
    end
end

%% Normalize the sum of all voxels to one inside the cord
tract_sum = zeros(size(tracts{1}));
tracts_norm = tracts;
% sum all labels
for ilabel = 1:length(tracts)
    tract_sum = tract_sum + tracts{ilabel};
end
imagesc(tract_sum(:,:,nb_slices)), axis square, title('tract sum'), colorbar
% smooth with 3d isotropic kernel
tract_sum_smooth = smooth3(tract_sum,'gaussian',[3 3 3]);
imagesc(tract_sum_smooth(:,:,nb_slices)), axis square, title('tract sum smooth'), colorbar
% binarize
tract_sum_smooth(tract_sum_smooth<0.5) = 0;
tract_sum_smooth(tract_sum_smooth>=0.5) = 1;
imagesc(tract_sum_smooth(:,:,nb_slices)), axis square, title('tract sum smooth binary'), colorbar
% get mask of non-null voxels
nonnull = find(tract_sum_smooth);
% loop across all labels and normalize to one
for ilabel = 1:length(label_values)
    tracts_norm{ilabel}(nonnull) = tracts{ilabel}(nonnull) ./ tract_sum(nonnull);
end
% sanity check
tract_sum_norm = zeros(size(tracts{1}));
for ilabel = 1:length(tracts)
    tract_sum_norm = tract_sum_norm + tracts_norm{ilabel};
end
imagesc(tract_sum_norm(:,:,nb_slices)), axis square, title('tract norm sum'), colorbar
% update variable
tracts = tracts_norm

%% Save tracts to file
disp('*** Save tracts to file ***')
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

%% Build global WM atlas
description = {'WM'};
global_suffix = '_wm';
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

%% Build global GM atlas
description = {'GM'};
global_suffix = '_gm';
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

%% Build global cord atlas
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
    numlabel = sprintf('%02d',str2num(struct_label{label}.id));
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


%% FINISHED!
bricon = ' -b 0.5,1 '
disp 'Done! To see results, type:'
disp(['cd ',path_out folder_final])
disp(['fslview ',path_template,'../../../data/',which_template,'/template/',which_template,'_t2.nii.gz -b 0,4000 PAM50_wm.nii.gz',bricon,' -l Blue-Lightblue ',prefix_out,'_00.nii.gz -l Red',bricon,prefix_out,'_01.nii.gz -l Green',bricon,prefix_out,'_02.nii.gz -l Blue',bricon,prefix_out,'_03.nii.gz -l Yellow',bricon,prefix_out,'_04.nii.gz -l Pink ',bricon,prefix_out,'_30.nii.gz -l Red-Yellow ',bricon,prefix_out,'_31.nii.gz -l Copper ',bricon,' &'])
