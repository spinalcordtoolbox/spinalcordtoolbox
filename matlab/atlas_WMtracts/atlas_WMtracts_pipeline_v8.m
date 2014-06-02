
%-------------- White matter tracts template construction -----------------
%--------------------------------------------------------------------------

% This script is made to construct a partial volume white
% matter tracts template, using raw anatomic atlas information which
% contains the white matter tracts and a spinal cord template (or data) on
% which we want to intergrate information on the white matter tracts

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
% Other dependencies: FSL, c3d, ANTs



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

% USER PARAMETER
% put the path to the template. Put "/" at the end
path_template = '/home/django/mbenhamou/matlab/toolbox/spinalcordtoolbox_dev/data/template/';
% name of the WM template. Default is 'MNI-Poly-AMU_WM'
file_template = 'MNI-Poly-AMU_WM';

% path to the image file that contains the drawing of the WM atlas, e.g., Gray's anatomy. Put "/" at the end
path_atlas_data = '/home/django/mbenhamou/matlab/toolbox/spinalcordtoolbox_dev/data/atlas/raw_data/';
% file name of the full atlas
file_atlas = 'atlas_grays_cerv_sym_correc_r2';
% file name of the binary mask that helps for the registration to the MNI-Poly-AMU
file_mask = 'mask_grays_cerv_sym_correc_r2';
ext_atlas = '.png';

% corresponds to mid-C4 in the MNI-Poly-AMU template
z_slice_ref = 411;
% interpolation factor for the MNI-Poly-AMU template in order to match the hi-res gray's atlas
interp_factor = 12;
% values of the label in the atlas file (file_atlas). Each value corresponds to a given tract, e.g., corticospinal left.
label_values = [14 26 38 47 52 62 70 82 89 94 101 107 112 116 121 146 152 159 167 173 180 187 194 199 204 208 214 219 224 230];
% these are the value corresponding to the slice number (z) on the MNI-Poly-AMU template, at which the atlas will be warped. It corresponds to the mid-levels as well as the level of the intervertebral disks.
% NB: to extract these values, you have to look at the T2 and WM tamplate, because this script will crop the WM template (which can be smaller than the T2), therefore the maximum z cannot exceed the zmax that will be generated in the cropped version of the WM template.
z_disks_mid = [506 498 481 464 446 429 411 393 376 359 342 325 306 287 267 247 225 204 183 162 140 118 96 74 51 29 14 0];
% same as before-- except that C4 mid-vertebral is not listed. 
z_disks_mid_noC4 = [506 498 481 464 446 429 393 376 359 342 325 306 287 267 247 225 204 183 162 140 118 96 74 51 29 14 0];



%--------------------------------------------------------------------------
%----------------------- Starting the pipeline -------------------------
%--------------------------------------------------------------------------

ext = '.nii.gz';
path_out = 'WMtracts_outputs/';
path_ctrl = [path_out 'registered_template/'];
path_results = [path_out 'final_results/'];
mkdir(path_out);
mkdir(path_ctrl);
mkdir(path_results);
prefix_out = 'WMtract_';

template_mask = [path_out file_template '_mask'];
template_cropped = [path_out file_template '_c6v'];
template_cropped_interp = [template_cropped '_int' num2str(interp_factor)];
template_cig = [template_cropped '_intgauss' num2str(interp_factor)];
templateci_slice_ref = [template_cropped_interp '_slice_ref'];
templateci_slice_ref_thresh = [templateci_slice_ref '_thresh'];
templateci_sr_nohd = [templateci_slice_ref '_nohd'];
templateci_thresh = [template_cropped_interp '_thrp6'];
templateci_srt_nohd = [templateci_slice_ref_thresh '_nohd'];

tracts_atlas = cell(1,length(label_values));
mask_nohd = [path_out file_mask];
atlas_nifti = [path_out file_atlas];

num_slice_ref = z_slice_ref + 1;
perc_up = 100*interp_factor;
perc_dn = 100/interp_factor;
prefix_ants = [path_out 'reg_'];
prefix_ants_ref = [path_out 'reg_ref_'];
affine_atlas = [prefix_ants_ref 'Affine.txt'];
Warp_atlas = [prefix_ants 'Warp_init' ext];
Warp_tmp = [prefix_ants 'Warp_init'];
suffix_ants = '_reg';


%--------------------------------------------------------------------------
%--- Preliminary operations: cropping and interpolation of the template ---

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
cmd = ['c3d ' template_mask ext ' -trim 6vox -o ' template_cropped ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Interpolation of the template
cmd = ['c3d ' template_cropped ext ' -interpolation Linear -resample ',...
    num2str(perc_up) 'x' num2str(perc_up) 'x100% -o ' template_cropped_interp ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Gaussian interpolation for registration computation
cmd = ['c3d ' template_cropped ext ' -interpolation Gaussian 1vox -resample ',...
    num2str(perc_up) 'x' num2str(perc_up) 'x100% -o ' template_cig ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Get a version of ref slice without geometrical information
cmd = ['c3d ' template_cropped_interp ext ' -slice z ' num2str(z_slice_ref) ' -o ' templateci_slice_ref ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

[sliceref,~,scales] = read_avw(templateci_slice_ref);
sliceref = m_normalize(sliceref);
save_avw(sliceref,templateci_sr_nohd,'d',scales);

% Binarization of the reference slice for the registration of the atlas
cmd = ['c3d ' templateci_slice_ref ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_slice_ref_thresh ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Binarization of the template for slice coregistration
cmd = ['c3d ' template_cropped_interp ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_thresh ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Get a version of binarized ref slice without geometrical information
[sliceref,~,scales] = read_avw(templateci_slice_ref_thresh);
sliceref = m_normalize(sliceref);
save_avw(sliceref,templateci_srt_nohd,'d',scales);

% Save the atlas and mask into a nifti with the same scales as the template
% One separate file for each tract
[slice_ref,~,scales] = read_avw(templateci_srt_nohd);
atlas = imread([path_atlas_data file_atlas ext_atlas]);

for label = 1:length(label_values)
    temp = zeros(size(atlas));
    ind = find( atlas == label_values(label) );
    temp(ind) = 1;
    tracts_atlas{label} = temp;
    
    tract_atlas = [path_out 'tract_atlas_' num2str(label)];
    save_avw(tracts_atlas{label},tract_atlas,'d',scales);
end

atlas = m_normalize(atlas);
save_avw(atlas,atlas_nifti,'d',scales);

mask = imread([path_atlas_data file_mask ext_atlas]);
mask = m_normalize(mask);
save_avw(mask,mask_nohd,'d',scales);

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


% Registration transform computation
cmd = ['ants 2 -o ' prefix_ants_ref ' ',...
    '-m PSE[' templateci_slice_ref_thresh ext ',' mask_nohd ext ',' templateci_slice_ref_thresh ext ',' mask_nohd ext ',0.5,100,11,0,10,1000] ',...
    '-m MSQ[' templateci_slice_ref_thresh ext ',' mask_nohd ext ',0.5,0] ',...
    '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
    '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Rename warping field
movefile([prefix_ants_ref 'Warp.nii.gz'],[Warp_tmp ext]);

% Constraint the warping field to preserve symmetry
cmd = ['c3d -mcs ' Warp_tmp ext ' -oo ' Warp_tmp 'x' ext ' ' Warp_tmp 'y' ext];
disp(cmd)
[status,result] = unix(cmd);
if(status),error(result);end

[warpx,dims,scales] = read_avw([Warp_tmp 'x' ext]);
warpy = read_avw([Warp_tmp 'y' ext]);
warpx = (warpx - warpx(end:-1:1,:)) / 2;
warpy = (warpy + warpy(end:-1:1,:)) / 2;

save_avw(warpx,[Warp_tmp 'xa' ext],'d',scales);
save_avw(warpy,[Warp_tmp 'ys' ext],'d',scales);

cmd = ['c3d ' Warp_tmp 'x' ext ' ' Warp_tmp 'xa' ext ' -copy-transform -o ' Warp_tmp 'xa' ext];
disp(cmd)
[status,result] = unix(cmd);
if(status),error(result);end

cmd = ['c3d ' Warp_tmp 'y' ext ' ' Warp_tmp 'ys' ext ' -copy-transform -o ' Warp_tmp 'ys' ext];
disp(cmd)
[status,result] = unix(cmd);
if(status),error(result);end

cmd = ['c3d ' Warp_tmp 'xa' ext ' ' Warp_tmp 'ys' ext ' -omc 2 ' Warp_atlas];
disp(cmd)
[status,result] = unix(cmd);
if(status),error(result);end

% Applying tranform to the mask
cmd = ['WarpImageMultiTransform 2 ' mask_nohd ext ' ' mask_nohd suffix_ants ext ' ' Warp_atlas ' ' affine_atlas ' -R ' templateci_slice_ref_thresh ext];
disp(cmd)
[status,result] = unix(cmd);
if(status),error(result);end

% Applying tranform to the initial atlas
cmd = ['WarpImageMultiTransform 2 ' atlas_nifti ext ' ' atlas_nifti suffix_ants ext ' ' Warp_atlas ' ' affine_atlas ' -R ' templateci_slice_ref_thresh ext];
disp(cmd)
[status,result] = unix(cmd);
if(status),error(result);end

% Applying tranform to the tract files and copying geometry and saving
for label = 1:length(label_values)/2
    tract_atlas_g = [path_out 'tract_atlas_' num2str(label)];
    tract_atlas_d = [path_out 'tract_atlas_' num2str(label+length(label_values)/2)];
    
    cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ' Warp_atlas ' ' affine_atlas ' -R ' templateci_slice_ref_thresh ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ' Warp_atlas ' ' affine_atlas ' -R ' templateci_slice_ref_thresh ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    cmd = ['c3d ' templateci_slice_ref ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    cmd = ['c3d ' templateci_slice_ref ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    tract_reg_g = [path_out 'tract_atlas_' num2str(label) suffix_ants];
    temp_g = read_avw(tract_reg_g);
    
    tract_reg_d = [path_out 'tract_atlas_' num2str(label+length(label_values)/2) suffix_ants];
    temp_d = read_avw(tract_reg_d);
    
    % Eliminate isolated values
    for i = 2:size(temp_g,1)-1
        for j = 2:size(temp_g,2)-1
            test = (temp_g(i,j)==temp_g(i-1,j)) || (temp_g(i,j)==temp_g(i,j-1)) || (temp_g(i,j)==temp_g(i+1,j)) || (temp_g(i,j)==temp_g(i,j+1));
            if(~test)
                temp_g(i,j) = 0;
            end
        end
    end
    
    for i = 2:size(temp_d,1)-1
        for j = 2:size(temp_d,2)-1
            test = (temp_d(i,j)==temp_d(i-1,j)) || (temp_d(i,j)==temp_d(i,j-1)) || (temp_d(i,j)==temp_d(i+1,j)) || (temp_d(i,j)==temp_d(i,j+1));
            if(~test)
                temp_d(i,j) = 0;
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
    
    tractsHR{label}(:,:,num_slice_ref) = temp_g;
    tractsHR{label+length(label_values)/2}(:,:,num_slice_ref) = temp_d;
    
end



%--------------------------------------------------------------------------
%---------------------- Construction of the template ----------------------



templatecit_slice_ref = [templateci_thresh '_slice_ref'];
cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(z_slice_ref) ' -o ' templatecit_slice_ref ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

[mstatus,msg] = copyfile([templateci_slice_ref ext],[template_cropped_interp '_slice' z_slice_ref ext]);
if(~mstatus), error(msg); end

reference_slice = [template_cropped_interp '_slice' z_slice_ref];
thr_ref_slice = templatecit_slice_ref;



%---------- Registration of initial atlas on each vertebral level ---------

for zslice = z_disks_mid_noC4
    
    
    
    numSlice = zslice + 1;
    
    templateci_slice = [template_cropped_interp '_slice' num2str(zslice)];
    templatecig_slice = [template_cig '_slice' num2str(zslice)];
    templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    atlas_slice = [atlas_nifti suffix_ants num2str(zslice)];
    warp_slice = [prefix_ants 'Warp.nii.gz'];
    affine_slice = [prefix_ants 'Affine.txt'];
    warp_temp = [prefix_ants 'Warp'];
    
    cmd = ['c3d ' template_cropped_interp ext ' -slice z ' num2str(zslice) ' -o ' templateci_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    cmd = ['c3d ' template_cig ext ' -slice z ' num2str(zslice) ' -o ' templatecig_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(zslice) ' -o ' templatecit_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    
    %---------- Registration of reference slice on current slice ----------
    
    % Registration transform computation
    cmd = ['ants 2 -o ' prefix_ants ' ',...
        '-m PSE[' templatecit_slice ext ',' thr_ref_slice ext ',' templatecit_slice ext ',' thr_ref_slice ext ',0.3,100,11,0,10,1000] ',...
        '-m MSQ[' templatecit_slice ext ',' thr_ref_slice ext ',0.3,0] ',...
        '-m MI[' templatecig_slice ',' templatecig_slice ',0.4,32] ',...
        '--use-all-metrics-for-convergence 1 ',...
        '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
        '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
    disp(cmd)
    [statusANTS,resultANTS] = unix(cmd);
    if(statusANTS), error(resultANTS); end
    
    % Constraint the warping field to preserve symmetry
    cmd = ['c3d -mcs ' warp_temp ext ' -oo ' warp_temp 'x' ext ' ' warp_temp 'y' ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    [warpx,dims,scales] = read_avw([warp_temp 'x' ext]);
    warpy = read_avw([warp_temp 'y' ext]);
    warpx = (warpx - warpx(end:-1:1,:)) / 2;
    warpy = (warpy + warpy(end:-1:1,:)) / 2;
    
    save_avw(warpx,[warp_temp 'xa' ext],'d',scales);
    save_avw(warpy,[warp_temp 'ys' ext],'d',scales);
    
    cmd = ['c3d ' warp_temp 'x' ext ' ' warp_temp 'xa' ext ' -copy-transform -o ' warp_temp 'xa' ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    cmd = ['c3d ' warp_temp 'y' ext ' ' warp_temp 'ys' ext ' -copy-transform -o ' warp_temp 'ys' ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    cmd = ['c3d ' warp_temp 'xa' ext ' ' warp_temp 'ys' ext ' -omc 2 ' warp_slice];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    % Apply transform to reference slice as a control
    cmd = ['WarpImageMultiTransform 2 ' reference_slice ext ' ' reference_slice suffix_ants num2str(zslice) ext ' ' warp_slice ' ' affine_slice ' -R ' templateci_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    % Apply transform to the initial atlas as a control
    cmd = ['WarpImageMultiTransform 2 ' atlas_nifti ext ' ' atlas_slice ext ' ' warp_slice ' ' affine_slice ' ' Warp_atlas ' ' affine_atlas ' -R ' templateci_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    % Apply tranform to the tract files and constraint to be symmetric
    for label = 1:length(label_values)/2
        tract_atlas_g = [path_out 'tract_atlas_' num2str(label)];
        tract_atlas_d = [path_out 'tract_atlas_' num2str(label+length(label_values)/2)];
        
        cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ' warp_slice ' ' affine_slice ' ' Warp_atlas ' ' affine_atlas ' -R ' templateci_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ' warp_slice ' ' affine_slice ' ' Warp_atlas ' ' affine_atlas ' -R ' templateci_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        cmd = ['c3d ' templateci_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        cmd = ['c3d ' templateci_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        tract_reg_g = [path_out 'tract_atlas_' num2str(label) suffix_ants];
        temp_g = read_avw(tract_reg_g);
        
        tract_reg_d = [path_out 'tract_atlas_' num2str(label+length(label_values)/2) suffix_ants];
        temp_d = read_avw(tract_reg_d);
        
        % Eliminate isolated values
        for i = 2:size(temp_g,1)-1
            for j = 2:size(temp_g,2)-1
                test = (temp_g(i,j)==temp_g(i-1,j)) || (temp_g(i,j)==temp_g(i,j-1)) || (temp_g(i,j)==temp_g(i+1,j)) || (temp_g(i,j)==temp_g(i,j+1));
                if(~test)
                    temp_g(i,j) = 0;
                end
            end
        end
        
        for i = 2:size(temp_d,1)-1
            for j = 2:size(temp_d,2)-1
                test = (temp_d(i,j)==temp_d(i-1,j)) || (temp_d(i,j)==temp_d(i,j-1)) || (temp_d(i,j)==temp_d(i+1,j)) || (temp_d(i,j)==temp_d(i,j+1));
                if(~test)
                    temp_d(i,j) = 0;
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
        
        tractsHR{label}(:,:,numSlice) = temp_g;
        tractsHR{label+length(label_values)/2}(:,:,numSlice) = temp_d;
        
    end
    
    
    
    % Move control files to control folder
    reg_slice_current = [reference_slice suffix_ants num2str(zslice)];
    movefile([reg_slice_current ext],path_ctrl);
    
    movefile([atlas_slice ext],path_ctrl);
    
    % Remove auxiliary files
    cmd = ['rm ' templateci_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    cmd = ['rm ' templatecit_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    cmd = ['rm ' templatecig_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    
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

for label = 1:length(label_values)
    
    % Save ML version and copy geometry
    filetractML = [path_results prefix_out '_' num2str(label)];
    save_avw(tracts{label},filetractML,'d',scalesCROP);
    cmd = ['c3d ' template_cropped ext ' ' filetractML ext ' -copy-transform -o ' filetractML ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
	 % Reslice into native template space
	cmd = ['c3d ' path_template file_template ext ' ' filetractML ext ' -reslice-identity -o ' filetractML ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
	 
	 
end









