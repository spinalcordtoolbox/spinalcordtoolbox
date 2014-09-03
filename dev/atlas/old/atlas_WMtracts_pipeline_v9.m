
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% White matter tracts template construction %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This script is made to construct a partial volume white
% matter tracts template, using raw anatomic atlas information which
% contains the white matter tracts and a spinal cord template (or data) on
% which we want to intergrate information on the white matter tracts

% v9: new propagation scheme to allow multiple atlas images input

%----------------------------- Dependencies -------------------------------
% Matlab dependencies:
% - image processing toolbox functions
% - m_normalize.m : normalization function
% - dnsamplelin.m : function for downsampling by computing mean value for
%   each region
%
% Other dependencies: FSL, c3d, ANTs



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%----------------------------- Template base ------------------------------
file_template = 'amu_v3_atlasW_ROI_sym_2_mni_3dof';
z_vertebrae = [506 481 446 411 376 342 306 267 225 183 140 96 51 14];
% z_vertebrae = [513 506 481 446 411 376 342 306 267 225 183 140 96 51 14 0]; %with top and bottom
% z_vertebrae_noC4 = [513 506 481 446 376 342 306 267 225 183 140 96 51 14 0]; %without C4 
% z_disks = [513 498 464 429 393 359 325 287 247 204 162 118 74 29 0];
z_disks_mid = [513 506 498 481 464 446 429 411 393 376 359 342 325 306 287 267 247 225 204 183 162 140 118 96 74 51 29 14 0];
% z_disks_mid_noC4 = [513 506 498 481 464 446 429 393 376 359 342 325 306 287 267 247 225 204 183 162 140 118 96 74 51 29 14 0];
interp_factor = 12;


%----------------------------- Atlas data ---------------------------------
number_data_slices = 2;
file_atlas = cell(number_data_slices,1);
file_mask = cell(number_data_slices,1);
label_values = cell(number_data_slices,1);
z_slice_ref = cell(number_data_slices,1);

% C1
file_atlas{1} = 'atlas_C1_r2_templatep6_sym';
file_mask{1} = 'mask_C1_r2_templatep6_sym';
label_values{1} = [14 26 38 146 152 159];
z_slice_ref{1} = z_vertebrae(1); 

% C4
file_atlas{2} = 'atlas_grays_cerv_sym_correc_r2';
file_mask{2} = 'mask_grays_cerv_sym_correc_r2';
label_values{2} = [14 26 38 47 52 62 70 82 89 94 101 107 112 116 121 146 152 159 167 173 180 187 194 199 204 208 214 219 224 230];
z_slice_ref{2} = z_vertebrae(4); 

master_ref_indx = 2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Start of the pipeline %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ext = '.nii.gz';
ext_atlas = '.png';
ext_aff = '.txt';
path_out = 'WMtracts_results/';
path_ctrl = [path_out 'registered_template/'];
mkdir(path_out);
mkdir(path_ctrl);
prefix_out = 'WMtract_';

perc_up = 100*interp_factor;
perc_dn = 100/interp_factor;
prefix_ants = [path_out 'reg_'];
suffix_ants = '_reg';
coreg_iterations = 8;

template_cropped = [path_out file_template '_c6v'];
template_cropped_interp = [template_cropped '_int' num2str(interp_factor)];
templateci_thresh = [template_cropped_interp '_thrp6'];

numtracts = 1;
z_slice_ref_indx = 1:length(z_slice_ref);
z_disks_mid_noref = z_disks_mid;

templateci_slice_ref = cell(number_data_slices,1);
templateci_slice_ref_thresh = cell(number_data_slices,1);
templatecit_slice_ref = cell(number_data_slices,1);

atlas_nifti = cell(number_data_slices,1);
mask_atlas = cell(number_data_slices,1);
num_slice_ref = cell(number_data_slices,1);
suffix_ants_ref = cell(number_data_slices,1);

for ref_indx = 1:number_data_slices
    
    numtracts = max(numtracts,length(label_values{ref_indx}));
    z_slice_ref_indx(ref_indx) = z_slice_ref{ref_indx};
    ind = find(z_disks_mid_noref == z_slice_ref{ref_indx});
    z_disks_mid_noref = [z_disks_mid_noref(1:ind-1) z_disks_mid_noref(ind+1:end)];
    
    templateci_slice_ref{ref_indx} = [template_cropped_interp '_slice_ref_' num2str(z_slice_ref{ref_indx})];
    templateci_slice_ref_thresh{ref_indx} = [templateci_slice_ref{ref_indx} '_thresh'];
    templatecit_slice_ref{ref_indx} = [templateci_thresh '_sliceref' num2str(z_slice_ref{ref_indx})];
    
    atlas_nifti{ref_indx} = [path_out file_atlas{ref_indx}];
    mask_atlas{ref_indx} = [path_out file_mask{ref_indx}];
    num_slice_ref{ref_indx} = z_slice_ref{ref_indx} + 1;
    suffix_ants_ref{ref_indx} = ['_reg_ref' num2str(z_slice_ref{ref_indx})];
    
end

prefix_ants_ref = [path_out 'reg_ref_'];
affine_atlas = [prefix_ants_ref 'Affine.txt'];
Warp_atlas = [prefix_ants 'Warp_init' ext];
Warp_tmp = [prefix_ants 'Warp_init'];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Preliminary operations: cropping and interpolation of the template %%%

% Cropping the template
cmd = ['c3d ' file_template ext ' -trim 6vox -o ' template_cropped ext];
[status,result] = unix(cmd);
if(status), error(result); end

% Interpolation of the template
cmd = ['c3d ' template_cropped ext ' -interpolation Linear -resample ',...
    num2str(perc_up) 'x' num2str(perc_up) 'x100% -o ' template_cropped_interp ext];
[status,result] = unix(cmd);
if(status), error(result); end

% Binarization of the template for slice coregistration
cmd = ['c3d ' template_cropped_interp ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_thresh ext];
[status,result] = unix(cmd);
if(status), error(result); end


% Get the template reference slice associated with each atlas data
for ref_indx = 1:number_data_slices
    
    cmd = ['c3d ' template_cropped_interp ext ' -slice z ' num2str(z_slice_ref{ref_indx}) ' -o ' templateci_slice_ref{ref_indx} ext];
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    cmd = ['c3d ' templateci_slice_ref{ref_indx} ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_slice_ref_thresh{ref_indx} ext];
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    
end


% Save each atlas and mask into a nifti with the same scales as the template
% One separate file for each tract
for ref_indx = 1:number_data_slices
    
    [~,~,scales] = read_avw(templateci_slice_ref{ref_indx});
    atlas = imread([file_atlas{ref_indx} ext_atlas]);
    mask = imread([file_mask{ref_indx} ext_atlas]);
    mask = m_normalize(mask);
    
    for label = 1:length(label_values{ref_indx})
        temp = zeros(size(atlas));
        ind = find( atlas == label_values{ref_indx}(label) );
        temp(ind) = 1;
        
        tract_atlas = [path_out 'tract_atlas' num2str(z_slice_ref{ref_indx}) '_' num2str(label)];
        save_avw(temp,tract_atlas,'d',scales);
    end
    
    save_avw(atlas,atlas_nifti{ref_indx},'d',scales);
    save_avw(mask,mask_atlas{ref_indx},'d',scales);
    
end


% Initializing outputs
[templateCROP,dimsCROP,scalesCROP] = read_avw(template_cropped);
[templateCI,dimsCI,scalesCI] = read_avw(template_cropped_interp);

tractsHR = cell(numtracts,1);
tracts = cell(numtracts,1);

for label = 1:numtracts
    tractsHR{label} = zeros(size(templateCI));
    tracts{label} = zeros(size(templateCROP));
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Registration of atlas data into template space %%%%%%%%%%%%%


for ref_indx = 1:number_data_slices
    
    if(ref_indx == master_ref_indx)
        
        % Registration transform computation
        cmd = ['ants 2 -o ' prefix_ants ' ',...
            '-m PSE[' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{ref_indx} ext ',' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{ref_indx} ext ',0.5,100,11,0,10,1000] ',...
            '-m MSQ[' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{ref_indx} ext ',0.5,0] ',...
            '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
            '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
        [status,result] = unix(cmd);
        if(status), error(result); end
        
        % Constraint the warping field to preserve symmetry
        Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
        
        % Applying tranform to the mask
        cmd = ['WarpImageMultiTransform 2 ' mask_atlas{ref_indx} ext ' ' mask_atlas{ref_indx} suffix_ants_ref{ref_indx} ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        % Applying tranform to the initial atlas
        cmd = ['WarpImageMultiTransform 2 ' atlas_nifti{ref_indx} ext ' ' atlas_nifti{ref_indx} suffix_ants_ref{ref_indx} ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        
        % Applying tranform to the tract files and copying geometry and saving
        for label = 1:length(label_values{ref_indx})/2
            tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{ref_indx}) '_' num2str(label)];
            tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{ref_indx}) '_' num2str(label+length(label_values{ref_indx})/2)];
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants_ref{ref_indx} ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants_ref{ref_indx} ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templateci_slice_ref{ref_indx} ext ' ' tract_atlas_g suffix_ants_ref{ref_indx} ext ' -copy-transform -o ' tract_atlas_g suffix_ants_ref{ref_indx} ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templateci_slice_ref{ref_indx} ext ' ' tract_atlas_d suffix_ants_ref{ref_indx} ext ' -copy-transform -o ' tract_atlas_d suffix_ants_ref{ref_indx} ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            tract_reg_g = [tract_atlas_g suffix_ants_ref{ref_indx}];
            temp_g = read_avw(tract_reg_g);
            
            tract_reg_d = [tract_atlas_d suffix_ants_ref{ref_indx}];
            temp_d = read_avw(tract_reg_d);
            
            % Eliminate isolated values
            temp_g = m_clean_points(temp_g,0);
            temp_d = m_clean_points(temp_d,0);
            
            % Symmetry constraint for left and right tracts
            temp_sum = temp_g + temp_d;
            temp_sum_flip = temp_sum(end:-1:1,:);
            temp_sym = (temp_sum + temp_sum_flip) / 2;
            
            temp_g(1:end/2,:) = 0;
            temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
            temp_d(1:end/2,:) = temp_sym(1:end/2,:);
            temp_d(1+end/2:end,:) = 0;
            
            tractsHR{label}(:,:,num_slice_ref{ref_indx}) = temp_g;
            tractsHR{label+length(label_values{ref_indx})/2}(:,:,num_slice_ref{ref_indx}) = temp_d;
        end
        
        % Get slice from the thresholded template for later registration
        cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(z_slice_ref{ref_indx}) ' -o ' templatecit_slice_ref{ref_indx} ext];
        [status,result] = unix(cmd);
        if(status), error(result); end
        
        
    else % ref_index ~= master_ref_index
        
        affine_shape = [prefix_ants 'affine_shape'];
        warp_shape = [prefix_ants 'warp_shape'];
        atlas_slice_master_reg = [atlas_nifti{master_ref_indx} suffix_ants_ref{ref_indx} '_tmp'];
        atlas_slice_master_reg_partial = [atlas_slice_master_reg '_partial'];
        
        % Registration transform computation for current ref
        cmd = ['ants 2 -o ' prefix_ants ' ',...
            '-m PSE[' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{ref_indx} ext ',' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{ref_indx} ext ',0.5,100,11,0,10,1000] ',...
            '-m MSQ[' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{ref_indx} ext ',0.5,0] ',...
            '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
            '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
        [status,result] = unix(cmd);
        if(status), error(result); end
        
        % Constraint the warping field to preserve symmetry
        Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
        
        % Applying tranform to the mask
        cmd = ['WarpImageMultiTransform 2 ' mask_atlas{ref_indx} ext ' ' mask_atlas{ref_indx} suffix_ants_ref{ref_indx} ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        % Applying tranform to the initial atlas
        cmd = ['WarpImageMultiTransform 2 ' atlas_nifti{ref_indx} ext ' ' atlas_nifti{ref_indx} suffix_ants_ref{ref_indx} ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        % Registration transform computation for master ref
        cmd = ['ants 2 -o ' prefix_ants ' ',...
            '-m PSE[' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{master_ref_indx} ext ',' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{master_ref_indx} ext ',0.5,100,11,0,10,1000] ',...
            '-m MSQ[' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{master_ref_indx} ext ',0.5,0] ',...
            '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
            '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
        [status,result] = unix(cmd);
        if(status), error(result); end
        
        % Constraint the warping field to preserve symmetry
        Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
        
        % Apply transform to the master atlas
        cmd = ['WarpImageMultiTransform 2 ' atlas_nifti{master_ref_indx} ext ' ' atlas_nifti{master_ref_indx} suffix_ants_ref{ref_indx} '_tmp' ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        % Rename shaping transforms
        [scs,msg] = movefile([prefix_ants 'Affine' ext_aff],[affine_shape ext_aff]);
        if(~scs), error(msg); end
        [scs,msg] = movefile([Warp_sym ext],[warp_shape ext]);
        if(~scs), error(msg); end
        
        % Select in master slice relevant tracts for registration
        [atlas_slice_master_reg_ML,~,scales] = read_avw(atlas_slice_master_reg);
        values = sort([0 4 255 label_values{master_ref_indx}]);
        atlas_slice_master_reg_ML = m_quantify_image(atlas_slice_master_reg_ML,values);
        atlas_slice_master_reg_partial_ML = 255 * ones(size(atlas_slice_master_reg_ML));
        atlas_slice_master_reg_partial_ML (atlas_slice_master_reg_ML == 0) = 0;
        temp = read_avw(atlas_nifti{ref_indx});
        labels_slice = unique(temp);
        labels_slice = labels_slice(:)';
        labels_slice = labels_slice(labels_slice > 0);
        labels_slice = labels_slice(labels_slice < 255);
        for label = labels_slice
            atlas_slice_master_reg_partial_ML(atlas_slice_master_reg_ML == label) = label;
        end
        atlas_slice_master_reg_partial_ML = m_clean_points(atlas_slice_master_reg_partial_ML,255);
        save_avw(atlas_slice_master_reg_partial_ML,atlas_slice_master_reg_partial,'d',scales);
        cmd = ['c3d ' atlas_slice_master_reg ext ' ' atlas_slice_master_reg_partial ext ' -copy-transform -o ' atlas_slice_master_reg_partial ext];
        [status,result] = unix(cmd);
        if(status), error(result); end
        
        % Registration of the master atlas on the current atlas
        cmd = ['ants 2 -o ' prefix_ants ' ',...
            '-m PSE[' atlas_nifti{ref_indx} suffix_ants_ref{ref_indx} ext ',' atlas_slice_master_reg_partial ext ',' atlas_nifti{ref_indx} suffix_ants_ref{ref_indx} ext ',' atlas_slice_master_reg_partial ext ',0.5,100,11,0,10,1000] ',...
            '-m MSQ[' atlas_nifti{ref_indx} suffix_ants_ref{ref_indx} ext ',' atlas_slice_master_reg_partial ext ',0.5,0] ',...
            '--use-all-metrics-for-convergence 1 ',...
            '-t SyN[0.4] -r Gauss[2,1] -i 1000x500x300 ',...
            '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1x0x0'];
        [status,result] = unix(cmd);
        if(status), error(result); end
        
        % Constraint the warping field to preserve symmetry
        Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
        
        % Applying transform to the master atlas
        cmd = ['WarpImageMultiTransform 2 ' atlas_nifti{master_ref_indx} ext ' ' atlas_nifti{master_ref_indx} suffix_ants_ref{ref_indx} '_tmp2' ext ' ',...
            Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        % Applying tranform to the tract files and copying geometry and saving
        for label = 1:length(label_values{master_ref_indx})/2
            tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label)];
            tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label+length(label_values{master_ref_indx})/2)];
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants_ref{ref_indx} ext ' ',...
                Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants_ref{ref_indx} ext ' ',...
                Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templateci_slice_ref{ref_indx} ext ' ' tract_atlas_g suffix_ants_ref{ref_indx} ext ' -copy-transform -o ' tract_atlas_g suffix_ants_ref{ref_indx} ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templateci_slice_ref{ref_indx} ext ' ' tract_atlas_d suffix_ants_ref{ref_indx} ext ' -copy-transform -o ' tract_atlas_d suffix_ants_ref{ref_indx} ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            tract_reg_g = [tract_atlas_g suffix_ants_ref{ref_indx}];
            temp_g = read_avw(tract_reg_g);
            
            tract_reg_d = [tract_atlas_d suffix_ants_ref{ref_indx}];
            temp_d = read_avw(tract_reg_d);
            
            % Eliminate isolated values
            temp_g = m_clean_points(temp_g,0);
            temp_d = m_clean_points(temp_d,0);
            
            % Symmetry constraint for left and right tracts
            temp_sum = temp_g + temp_d;
            temp_sum_flip = temp_sum(end:-1:1,:);
            temp_sym = (temp_sum + temp_sum_flip) / 2;
            
            temp_g(1:end/2,:) = 0;
            temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
            temp_d(1:end/2,:) = temp_sym(1:end/2,:);
            temp_d(1+end/2:end,:) = 0;
            
            tractsHR{label}(:,:,num_slice_ref{ref_indx}) = temp_g;
            tractsHR{label+length(label_values{master_ref_indx})/2}(:,:,num_slice_ref{ref_indx}) = temp_d;
        end
        
        % Get slice from the thresholded template for later registration
        cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(z_slice_ref{ref_indx}) ' -o ' templatecit_slice_ref{ref_indx} ext];
        [status,result] = unix(cmd);
        if(status), error(result); end
        
    end
    
    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% Coregistration for intermediate vertebral levels %%%%%%%%%%%%


num_iter = 0;



for zslice = z_disks_mid_noref
    
    num_iter = num_iter + 1;
    num_slice = zslice + 1;
    
    templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    
    cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(zslice) ' -o ' templatecit_slice ext];
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    
    
    % ------------------------------ CASE 1 -------------------------------
    if (zslice > z_slice_ref{1}) 
        % Above first reference -- Only first ref and master ref
        
        
        % --------------------------- CASE 1.1 ----------------------------
        if (master_ref_indx == 1)
            
            sliceref_thresh_bottom = templatecit_slice_ref{1};
            atlas_slice = [atlas_nifti{1} suffix_ants_ref{1}];
            atlas_slice_reg = [atlas_slice suffix_ants num2str(zslice)];
            
            % Registration transform computation
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_bottom ext ',' templatecit_slice ext ',' sliceref_thresh_bottom ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_bottom ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the initial atlas as a control
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice ext ' ' atlas_slice_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            
            % Apply tranform to the tract files and constraint to be symmetric
            for label = 1:length(label_values{1})/2
                
                tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{1}) '_' num2str(label) suffix_ants_ref{1}];
                tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{1}) '_' num2str(label+length(label_values{master_ref_indx})/2) suffix_ants_ref{1}];
                
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ',...
                    Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ',...
                    Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                tract_reg_g = [tract_atlas_g suffix_ants];
                temp_g = read_avw(tract_reg_g);
                
                tract_reg_d = [tract_atlas_d suffix_ants];
                temp_d = read_avw(tract_reg_d);
                
                % Eliminate isolated values
                temp_g = m_clean_points(temp_g,0);
                temp_d = m_clean_points(temp_d,0);
                
                % Symmetry constraint for left and right tracts
                temp_sum = temp_g + temp_d;
                temp_sum_flip = temp_sum(end:-1:1,:);
                temp_sym = (temp_sum + temp_sum_flip) / 2;
                
                temp_g(1:end/2,:) = 0;
                temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
                temp_d(1:end/2,:) = temp_sym(1:end/2,:);
                temp_d(1+end/2:end,:) = 0;
                
                tractsHR{label}(:,:,num_slice) = temp_g;
                tractsHR{label+length(label_values{1})/2}(:,:,num_slice) = temp_d;
                
            end
            
            
            % Move control files to control folder
            [scs,msg] = movefile([atlas_slice_reg ext],path_ctrl);
            if(~scs), error(msg); end
            
            % Remove auxiliary files
            cmd = ['rm ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            
            % ------------------------- CASE 1.2 --------------------------
        else % master_ref_indx ~= 1
            
            
            sliceref_thresh_bottom = templatecit_slice_ref{1};
            sliceref_thresh_master = templatecit_slice_ref{master_ref_indx};
            atlas_slice = [atlas_nifti{1} suffix_ants_ref{1}];
            atlas_slice_master = [atlas_nifti{master_ref_indx} suffix_ants_ref{master_ref_indx}];
            atlas_slice_reg = [atlas_slice suffix_ants num2str(zslice)];
            atlas_slice_master_reg = [atlas_slice_master suffix_ants num2str(zslice)];
            atlas_slice_master_ctrl = [atlas_slice_master_reg '_ctrl'];
            atlas_slice_master_reg_partial = [atlas_slice_master_reg '_partial'];
            warp_shape = [prefix_ants 'warp_shape'];
            affine_shape = [prefix_ants 'affine_shape'];
            
            % Registration transform computation for top ref slice
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_bottom ext ',' templatecit_slice ext ',' sliceref_thresh_bottom ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_bottom ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the initial atlas
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice ext ' ' atlas_slice_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Registration transform computation for master ref slice
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_master ext ',' templatecit_slice ext ',' sliceref_thresh_master ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_master ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the master atlas
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_master ext ' ' atlas_slice_master_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Rename shaping transforms
            [scs,msg] = movefile([prefix_ants 'Affine' ext_aff],[affine_shape ext_aff]);
            if(~scs), error(msg); end
            [scs,msg] = movefile([Warp_sym ext],[warp_shape ext]);
            if(~scs), error(msg); end
            
            % Select in master slice relevant tracts for registration
            [atlas_slice_master_reg_ML,~,scales] = read_avw(atlas_slice_master_reg);
            values = sort([0 4 255 label_values{master_ref_indx}]);
            atlas_slice_master_reg_ML = m_quantify_image(atlas_slice_master_reg_ML,values);
            atlas_slice_master_reg_partial_ML = 255 * ones(size(atlas_slice_master_reg_ML));
            atlas_slice_master_reg_partial_ML (atlas_slice_master_reg_ML == 0) = 0;
            temp = read_avw(atlas_nifti{1});
            labels_slice = unique(temp);
            labels_slice = labels_slice(:)';
            labels_slice = labels_slice(labels_slice > 0);
            labels_slice = labels_slice(labels_slice < 255);
            for label = labels_slice
                atlas_slice_master_reg_partial_ML(atlas_slice_master_reg_ML == label) = label;
            end
            atlas_slice_master_reg_partial_ML = m_clean_points(atlas_slice_master_reg_partial_ML,255);
            save_avw(atlas_slice_master_reg_partial_ML,atlas_slice_master_reg_partial,'d',scales);
            cmd = ['c3d ' atlas_slice_master_reg ext ' ' atlas_slice_master_reg_partial ext ' -copy-transform -o ' atlas_slice_master_reg_partial ext];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            % Registration of master atlas on top ref atlas
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' atlas_slice_reg ext ',' atlas_slice_master_reg_partial ext ',' atlas_slice_reg ext ',' atlas_slice_master_reg_partial ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' atlas_slice_reg ext ',' atlas_slice_master_reg_partial ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.4] -r Gauss[2,1] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1x0x0'];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply to atlas as a control
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_master ext ' ' atlas_slice_master_ctrl ext ' ',...
                Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Apply tranform to the tract files and constraint to be symmetric
            for label = 1:length(label_values{master_ref_indx})/2
                
                tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label) suffix_ants_ref{master_ref_indx}];
                tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label+length(label_values{master_ref_indx})/2) suffix_ants_ref{master_ref_indx}];
                
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ',...
                    Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ',...
                    Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                tract_reg_g = [tract_atlas_g suffix_ants];
                temp_g = read_avw(tract_reg_g);
                
                tract_reg_d = [tract_atlas_d suffix_ants];
                temp_d = read_avw(tract_reg_d);
                
                % Eliminate isolated values
                temp_g = m_clean_points(temp_g,0);
                temp_d = m_clean_points(temp_d,0);
                
                % Symmetry constraint for left and right tracts
                temp_sum = temp_g + temp_d;
                temp_sum_flip = temp_sum(end:-1:1,:);
                temp_sym = (temp_sum + temp_sum_flip) / 2;
                
                temp_g(1:end/2,:) = 0;
                temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
                temp_d(1:end/2,:) = temp_sym(1:end/2,:);
                temp_d(1+end/2:end,:) = 0;
                
                tractsHR{label}(:,:,num_slice) = temp_g;
                tractsHR{label+length(label_values{master_ref_indx})/2}(:,:,num_slice) = temp_d;
                
            end
            
            
            % Move control files to control folder
            [scs,msg] = movefile([atlas_slice_master_ctrl ext],path_ctrl);
            if(~scs), error(msg); end
            
            % Remove auxiliary files
            cmd = ['rm ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            
            
        end % if master_ref_indx == 1
        %------------------------------------------------------------------
        
        
        
        
        
        % ---------------------------- CASE 2 -----------------------------
    elseif (zslice < z_slice_ref{end}) 
        % Below last reference -- Only last ref and master ref
        
        
        % --------------------------- CASE 2.1 ----------------------------
        if (master_ref_indx == number_data_slices)
            
            
            sliceref_thresh_top = templatecit_slice_ref{end};
            atlas_slice = [atlas_nifti{end} suffix_ants_ref{end}];
            atlas_slice_reg = [atlas_slice suffix_ants num2str(zslice)];
            
            % Registration transform computation
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_top ext ',' templatecit_slice ext ',' sliceref_thresh_top ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_top ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the initial atlas as a control
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice ext ' ' atlas_slice_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Apply tranform to the tract files and constraint to be symmetric
            for label = 1:length(label_values{end})/2
                
                tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{end}) '_' num2str(label) suffix_ants_ref{end}];
                tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{end}) '_' num2str(label+length(label_values{master_ref_indx})/2) suffix_ants_ref{end}];
                
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ',...
                    Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ',...
                    Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                tract_reg_g = [tract_atlas_g suffix_ants];
                temp_g = read_avw(tract_reg_g);
                
                tract_reg_d = [tract_atlas_d suffix_ants];
                temp_d = read_avw(tract_reg_d);
                
                % Eliminate isolated values
                temp_g = m_clean_points(temp_g,0);
                temp_d = m_clean_points(temp_d,0);
                
                % Symmetry constraint for left and right tracts
                temp_sum = temp_g + temp_d;
                temp_sum_flip = temp_sum(end:-1:1,:);
                temp_sym = (temp_sum + temp_sum_flip) / 2;
                
                temp_g(1:end/2,:) = 0;
                temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
                temp_d(1:end/2,:) = temp_sym(1:end/2,:);
                temp_d(1+end/2:end,:) = 0;
                
                tractsHR{label}(:,:,num_slice) = temp_g;
                tractsHR{label+length(label_values{end})/2}(:,:,num_slice) = temp_d;
                
            end
            
            
            % Move control files to control folder
            [scs,msg] = movefile([atlas_slice_reg ext],path_ctrl);
            if(~scs), error(msg); end
            
            % Remove auxiliary files
            cmd = ['rm ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            
            % ------------------------- CASE 2.2 --------------------------
        else % master_ref_indx ~= number_data_slices
            
            
            sliceref_thresh_top = templatecit_slice_ref{end};
            sliceref_thresh_master = templatecit_slice_ref{master_ref_indx};
            atlas_slice = [atlas_nifti{end} suffix_ants_ref{end}];
            atlas_slice_master = [atlas_nifti{master_ref_indx} suffix_ants_ref{master_ref_indx}];
            atlas_slice_reg = [atlas_slice suffix_ants num2str(zslice)];
            atlas_slice_master_reg = [atlas_slice_master suffix_ants num2str(zslice)];
            atlas_slice_master_ctrl = [atlas_slice_master_reg '_ctrl'];
            atlas_slice_master_reg_partial = [atlas_slice_master_reg '_partial'];
            warp_shape = [prefix_ants 'warp_shape'];
            affine_shape = [prefix_ants 'affine_shape'];
            
            % Registration transform computation for top ref slice
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_top ext ',' templatecit_slice ext ',' sliceref_thresh_top ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_top ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the initial atlas
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice ext ' ' atlas_slice_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Registration transform computation for master ref slice
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_master ext ',' templatecit_slice ext ',' sliceref_thresh_master ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_master ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the master atlas
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_master ext ' ' atlas_slice_master_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Rename shaping transforms
            [scs,msg] = movefile([prefix_ants 'Affine' ext_aff],[affine_shape ext_aff]);
            if(~scs), error(msg); end
            [scs,msg] = movefile([Warp_sym ext],[warp_shape ext]);
            if(~scs), error(msg); end
            
            % Select in master slice relevant tracts for registration
            [atlas_slice_master_reg_ML,~,scales] = read_avw(atlas_slice_master_reg);
            values = sort([0 4 255 label_values{master_ref_indx}]);
            atlas_slice_master_reg_ML = m_quantify_image(atlas_slice_master_reg_ML,values);
            atlas_slice_master_reg_partial_ML = 255 * ones(size(atlas_slice_master_reg_ML));
            atlas_slice_master_reg_partial_ML (atlas_slice_master_reg_ML == 0) = 0;
            temp = read_avw(atlas_nifti{end});
            labels_slice = unique(temp);
            labels_slice = labels_slice(:)';
            labels_slice = labels_slice(labels_slice > 0);
            labels_slice = labels_slice(labels_slice < 255);
            for label = labels_slice
                atlas_slice_master_reg_partial_ML(atlas_slice_master_reg_ML == label) = label;
            end
            atlas_slice_master_reg_partial_ML = m_clean_points(atlas_slice_master_reg_partial_ML,255);
            save_avw(atlas_slice_master_reg_partial_ML,atlas_slice_master_reg_partial,'d',scales);
            cmd = ['c3d ' atlas_slice_master_reg ext ' ' atlas_slice_master_reg_partial ext ' -copy-transform -o ' atlas_slice_master_reg_partial ext];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            % Registration of master atlas on bottom ref atlas
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' atlas_slice_reg ext ',' atlas_slice_master_reg_partial ext ',' atlas_slice_reg ext ',' atlas_slice_master_reg_partial ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' atlas_slice_reg ext ',' atlas_slice_master_reg_partial ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.4] -r Gauss[2,1] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1x0x0'];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply to atlas as a control
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_master ext ' ' atlas_slice_master_ctrl ext ' ',...
                Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Apply tranform to the tract files and constraint to be symmetric
            for label = 1:length(label_values{master_ref_indx})/2
                
                tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label) suffix_ants_ref{master_ref_indx}];
                tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label+length(label_values{master_ref_indx})/2) suffix_ants_ref{master_ref_indx}];
                
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ',...
                    Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ',...
                    Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                tract_reg_g = [tract_atlas_g suffix_ants];
                temp_g = read_avw(tract_reg_g);
                
                tract_reg_d = [tract_atlas_d suffix_ants];
                temp_d = read_avw(tract_reg_d);
                
                % Eliminate isolated values
                temp_g = m_clean_points(temp_g,0);
                temp_d = m_clean_points(temp_d,0);
                
                % Symmetry constraint for left and right tracts
                temp_sum = temp_g + temp_d;
                temp_sum_flip = temp_sum(end:-1:1,:);
                temp_sym = (temp_sum + temp_sum_flip) / 2;
                
                temp_g(1:end/2,:) = 0;
                temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
                temp_d(1:end/2,:) = temp_sym(1:end/2,:);
                temp_d(1+end/2:end,:) = 0;
                
                tractsHR{label}(:,:,num_slice) = temp_g;
                tractsHR{label+length(label_values{master_ref_indx})/2}(:,:,num_slice) = temp_d;
                
            end
            
            
            % Move control files to control folder
            [scs,msg] = movefile([atlas_slice_master_ctrl ext],path_ctrl);
            if(~scs), error(msg); end
            
            % Remove auxiliary files
            cmd = ['rm ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            
            
        end % if master_ref_indx == number_data_slices
        %------------------------------------------------------------------
        
        
        
        
        % ---------------------------- CASE 3 -----------------------------
    else % zslice between references 
        % Intermediate slice -- Requires a coregistration of references
        
        indx_ref_bottom = 1;
        while (zslice < z_slice_ref{indx_ref_bottom})
            indx_ref_bottom = indx_ref_bottom + 1;
        end
        indx_ref_top = indx_ref_bottom - 1;
        
        sliceref_thresh_top = templatecit_slice_ref{indx_ref_top};
        sliceref_thresh_bottom = templatecit_slice_ref{indx_ref_bottom};
        
        
        % --------------------------- CASE 3.1 ----------------------------
        if (master_ref_indx == indx_ref_bottom || master_ref_indx == indx_ref_top)
            
            
            if (master_ref_indx == indx_ref_bottom)
                sliceref_thresh_master = sliceref_thresh_bottom;
                sliceref_thresh_comp = sliceref_thresh_top;
                indx_ref_comp = indx_ref_top;
            else
                sliceref_thresh_master = sliceref_thresh_top;
                sliceref_thresh_comp = sliceref_thresh_bottom;
                indx_ref_comp = indx_ref_bottom;
            end
            
            
            atlas_slice_master = [atlas_nifti{master_ref_indx} suffix_ants_ref{master_ref_indx}];
            atlas_slice_master_reg = [atlas_slice_master suffix_ants num2str(zslice)];
            atlas_slice_master_ctrl = [atlas_slice_master_reg '_ctrl'];
            atlas_slice_comp = [atlas_nifti{indx_ref_comp} suffix_ants_ref{indx_ref_comp}];
            atlas_slice_comp_reg = [atlas_slice_comp suffix_ants num2str(zslice)];
            atlas_slice_master_reg_partial = [atlas_slice_master_reg '_partial'];
            warp_shape = [prefix_ants 'warp_shape'];
            affine_shape = [prefix_ants 'affine_shape'];
            
            
            % Registration transform computation for complementary ref slice
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_comp ext ',' templatecit_slice ext ',' sliceref_thresh_comp ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_comp ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the initial atlas
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_comp ext ' ' atlas_slice_comp_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Registration transform computation for master ref slice
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_master ext ',' templatecit_slice ext ',' sliceref_thresh_master ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_master ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the master atlas
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_master ext ' ' atlas_slice_master_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Rename shaping transforms
            [scs,msg] = movefile([prefix_ants 'Affine' ext_aff],[affine_shape ext_aff]);
            if(~scs), error(msg); end
            [scs,msg] = movefile([Warp_sym ext],[warp_shape ext]);
            if(~scs), error(msg); end
            
            % Select in master slice relevant tracts for registration
            [atlas_slice_master_reg_ML,~,scales] = read_avw(atlas_slice_master_reg);
            values = sort([0 4 255 label_values{master_ref_indx}]);
            atlas_slice_master_reg_ML = m_quantify_image(atlas_slice_master_reg_ML,values);
            atlas_slice_master_reg_partial_ML = 255 * ones(size(atlas_slice_master_reg_ML));
            atlas_slice_master_reg_partial_ML (atlas_slice_master_reg_ML == 0) = 0;
            temp = read_avw(atlas_nifti{indx_ref_comp});
            labels_slice = unique(temp);
            labels_slice = labels_slice(:)';
            labels_slice = labels_slice(labels_slice > 0);
            labels_slice = labels_slice(labels_slice < 255);
            for label = labels_slice
                atlas_slice_master_reg_partial_ML(atlas_slice_master_reg_ML == label) = label;
            end
            atlas_slice_master_reg_partial_ML = m_clean_points(atlas_slice_master_reg_partial_ML,255);
            save_avw(atlas_slice_master_reg_partial_ML,atlas_slice_master_reg_partial,'d',scales);
            cmd = ['c3d ' atlas_slice_master_reg ext ' ' atlas_slice_master_reg_partial ext ' -copy-transform -o ' atlas_slice_master_reg_partial ext];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            % Coregistration between master and complementary slices
            [mean_reg,warps_m2ref,warps_c2ref] = m_coregister_multilabel(atlas_slice_master_reg_partial,atlas_slice_comp_reg,ext,0.5,0,coreg_iterations,1);
            
            % Apply to atlas as a control
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_master ext ' ' atlas_slice_master_ctrl ext ' ',...
                warps_m2ref{1} ext ' ' warps_m2ref{2} ext ' ' warps_m2ref{3} ext ' ' warps_m2ref{4} ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Apply tranform to the tract files and constraint to be symmetric
            for label = 1:length(label_values{master_ref_indx})/2
                
                tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label) suffix_ants_ref{master_ref_indx}];
                tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label+length(label_values{master_ref_indx})/2) suffix_ants_ref{master_ref_indx}];
                
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ',...
                    warps_m2ref{1} ext ' ' warps_m2ref{2} ext ' ' warps_m2ref{3} ext ' ' warps_m2ref{4} ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ',...
                    warps_m2ref{1} ext ' ' warps_m2ref{2} ext ' ' warps_m2ref{3} ext ' ' warps_m2ref{4} ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                tract_reg_g = [tract_atlas_g suffix_ants];
                temp_g = read_avw(tract_reg_g);
                
                tract_reg_d = [tract_atlas_d suffix_ants];
                temp_d = read_avw(tract_reg_d);
                
                % Eliminate isolated values
                temp_g = m_clean_points(temp_g,0);
                temp_d = m_clean_points(temp_d,0);
                
                % Symmetry constraint for left and right tracts
                temp_sum = temp_g + temp_d;
                temp_sum_flip = temp_sum(end:-1:1,:);
                temp_sym = (temp_sum + temp_sum_flip) / 2;
                
                temp_g(1:end/2,:) = 0;
                temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
                temp_d(1:end/2,:) = temp_sym(1:end/2,:);
                temp_d(1+end/2:end,:) = 0;
                
                tractsHR{label}(:,:,num_slice) = temp_g;
                tractsHR{label+length(label_values{master_ref_indx})/2}(:,:,num_slice) = temp_d;
                
            end
            
            
            % Move control files to control folder
            [scs,msg] = movefile([atlas_slice_master_ctrl ext],path_ctrl);
            if(~scs), error(msg); end
            
            % Remove auxiliary files
            cmd = ['rm ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            
            
            % ------------------------- CASE 3.2 --------------------------
        else % master_ref_indx ~= indx_ref_bottom & master_ref_indx ~= indx_ref_top
            
            
            sliceref_thresh_master = templatecit_slice_ref{master_ref_indx};
            atlas_slice_master = [atlas_nifti{master_ref_indx} suffix_ants_ref{master_ref_indx}];
            atlas_slice_master_reg = [atlas_slice_master suffix_ants num2str(zslice)];
            atlas_slice_master_ctrl = [atlas_slice_master_reg '_ctrl'];
            atlas_slice_master_reg_partial = [atlas_slice_master_reg '_partial'];
            atlas_slice_top = [atlas_nifti{indx_ref_top} suffix_ants_ref{indx_ref_top}];
            atlas_slice_top_reg = [atlas_slice_top suffix_ants num2str(zslice)];
            atlas_slice_bottom = [atlas_nifti{indx_ref_bottom} suffix_ants_ref{indx_ref_bottom}];
            atlas_slice_bottom_reg = [atlas_slice_bottom suffix_ants num2str(zslice)];
            warp_shape = [prefix_ants 'warp_shape'];
            affine_shape = [prefix_ants 'affine_shape'];
            
            
            % Registration transform computation for top ref slice
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_top ext ',' templatecit_slice ext ',' sliceref_thresh_top ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_top ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the initial atlas
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_top ext ' ' atlas_slice_top_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Registration transform computation for bottom ref slice
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_bottom ext ',' templatecit_slice ext ',' sliceref_thresh_bottom ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_bottom ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the initial atlas
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_bottom ext ' ' atlas_slice_bottom_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Coregistration between top and bottom references
            [mean_reg,warps_top2ref,warps_bot2ref] = m_coregister_multilabel(atlas_slice_top_reg,atlas_slice_bottom_reg,ext,0.5,0,coreg_iterations,1);
            
            % Registration transform computation for master ref slice
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' templatecit_slice ext ',' sliceref_thresh_master ext ',' templatecit_slice ext ',' sliceref_thresh_master ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_master ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply transform to the master atlas
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_master ext ' ' atlas_slice_master_reg ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Rename shaping transforms
            [scs,msg] = movefile([prefix_ants 'Affine' ext_aff],[affine_shape ext_aff]);
            if(~scs), error(msg); end
            [scs,msg] = movefile([Warp_sym ext],[warp_shape ext]);
            if(~scs), error(msg); end
            
            % Select in master slice relevant tracts for registration
            [atlas_slice_master_reg_ML,~,scales] = read_avw(atlas_slice_master_reg);
            values = sort([0 4 255 label_values{master_ref_indx}]);
            atlas_slice_master_reg_ML = m_quantify_image(atlas_slice_master_reg_ML,values);
            atlas_slice_master_reg_partial_ML = 255 * ones(size(atlas_slice_master_reg_ML));
            atlas_slice_master_reg_partial_ML (atlas_slice_master_reg_ML == 0) = 0;
            temp = read_avw(atlas_nifti{indx_ref_top});
            labels_slice = unique(temp);
            labels_slice = labels_slice(:)';
            labels_slice = labels_slice(labels_slice > 0);
            labels_slice = labels_slice(labels_slice < 255);
            for label = labels_slice
                atlas_slice_master_reg_partial_ML(atlas_slice_master_reg_ML == label) = label;
            end
            atlas_slice_master_reg_partial_ML = m_clean_points(atlas_slice_master_reg_partial_ML,255);
            save_avw(atlas_slice_master_reg_partial_ML,atlas_slice_master_reg_partial,'d',scales);
            cmd = ['c3d ' atlas_slice_master_reg ext ' ' atlas_slice_master_reg_partial ext ' -copy-transform -o ' atlas_slice_master_reg_partial ext];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            % Registration of master slice on mean coregistration of top
            % and bottom references
            cmd = ['ants 2 -o ' prefix_ants ' ',...
                '-m PSE[' mean_reg ext ',' atlas_slice_master_reg_partial ext ',' mean_reg ext ',' atlas_slice_master_reg_partial ext ',0.5,100,11,0,10,1000] ',...
                '-m MSQ[' mean_reg ext ',' atlas_slice_master_reg_partial ext ',0.5,0] ',...
                '--use-all-metrics-for-convergence 1 ',...
                '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
                '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
            [statusANTS,resultANTS] = unix(cmd);
            if(statusANTS), error(resultANTS); end
            
            % Constraint the warping field to preserve symmetry
            Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
            
            % Apply to atlas as a control
            cmd = ['WarpImageMultiTransform 2 ' atlas_slice_master ext ' ' atlas_slice_master_ctrl ext ' ',...
                Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            % Apply tranform to the tract files and constraint to be symmetric
            for label = 1:length(label_values{master_ref_indx})/2
                
                tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label) suffix_ants_ref{master_ref_indx}];
                tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{master_ref_indx}) '_' num2str(label+length(label_values{master_ref_indx})/2) suffix_ants_ref{master_ref_indx}];
                
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ',...
                    Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ',...
                    Warp_sym ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
                [status,result] = unix(cmd);
                if(status),error(result);end
                
                tract_reg_g = [tract_atlas_g suffix_ants];
                temp_g = read_avw(tract_reg_g);
                
                tract_reg_d = [tract_atlas_d suffix_ants];
                temp_d = read_avw(tract_reg_d);
                
                % Eliminate isolated values
                temp_g = m_clean_points(temp_g,0);
                temp_d = m_clean_points(temp_d,0);
                
                % Symmetry constraint for left and right tracts
                temp_sum = temp_g + temp_d;
                temp_sum_flip = temp_sum(end:-1:1,:);
                temp_sym = (temp_sum + temp_sum_flip) / 2;
                
                temp_g(1:end/2,:) = 0;
                temp_g(1+end/2:end,:) = temp_sym(1+end/2:end,:);
                temp_d(1:end/2,:) = temp_sym(1:end/2,:);
                temp_d(1+end/2:end,:) = 0;
                
                tractsHR{label}(:,:,num_slice) = temp_g;
                tractsHR{label+length(label_values{master_ref_indx})/2}(:,:,num_slice) = temp_d;
                
            end
            
            
            % Move control files to control folder
            movefile([atlas_slice_master_ctrl ext],path_ctrl);
            
            % Remove auxiliary files
            cmd = ['rm ' templatecit_slice ext ' '];
            [status,result] = unix(cmd);
            if(status), error(result); end
            
            
            
        end % if master_ref_indx
        %------------------------------------------------------------------
        
    end % if zslice position
    
    
    
end % for zslice







%%%%%%%%%%%%%%%%%% Interpolation between computed slices %%%%%%%%%%%%%%%%%%

for label = 1:length(label_values{master_ref_indx})
    for k = 1:length(z_disks_mid)-1
    
        tractsHR{label} = m_linear_interp(tractsHR{label},z_disks_mid(k)+1,z_disks_mid(k+1)+1);
%         tractsHR{label}(tractsHR{label} > 0.5) = 1; % ?
%         tractsHR{label}(tractsHR{label} < 0.5) = 0; % ?
        
    end
end




%%%%%%%%%%%%%%% Downsampling and partial volume computation %%%%%%%%%%%%%%%

for label = 1:length(label_values{master_ref_indx})
    for zslice = 0:513
        
        num_slice = zslice+1;
        
        tracts{label}(:,:,num_slice) = dnsamplelin(tractsHR{label}(:,:,num_slice),interp_factor);
        
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%% Writing the outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%

for label = 1:length(label_values{master_ref_indx})
    
    % Save ML version and copy geometry
    filetractML = [path_out prefix_out 'MLb_' num2str(label)];
    save_avw(tracts{label},filetractML,'d',scalesCROP);
    cmd = ['c3d ' template_cropped ext ' ' filetractML ext ' -copy-transform -o ' filetractML ext];
    [status,result] = unix(cmd);
    if(status), error(result); end
    
end









