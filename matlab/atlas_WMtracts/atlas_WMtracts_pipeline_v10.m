
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% White matter tracts template construction %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This script is made to construct a partial volume white
% matter tracts template, using raw anatomic atlas information which
% contains the white matter tracts and a spinal cord template (or data) on
% which we want to intergrate information on the white matter tracts

% v10: new registration scheme to improve accuracy of co-registration
%       between atlases

%----------------------------- Dependencies -------------------------------
% Matlab dependencies:
% - image processing toolbox functions
% - m_normalize.m : normalization function
% - dnsamplelin.m : function for downsampling by computing mean value for
%   each region
%
% Other dependencies: c3d, ANTs



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
z_disks_mid = [513 506 498 481 464 446 429 411 393 376 359 342 325 306 287 267 247 225 204 183 162 140 118 96 74 51 29 14 0];
interp_factor = 12;


%----------------------------- Atlas data ---------------------------------
number_data_slices = 2;
file_atlas = cell(number_data_slices,1);
file_mask = cell(number_data_slices,1);
label_values = cell(number_data_slices,1);
z_slice_ref = cell(number_data_slices,1);

% C1
file_atlas{1} = 'atlas_C1_r2_templatep6_sym'; % PNG image
file_mask{1} = 'mask_C1_r2_templatep6_sym'; % PNG image
label_values{1} = [14 26 38 146 152 159]; % PNG values [0 255]
z_slice_ref{1} = z_vertebrae(1); 

% C4
file_atlas{2} = 'atlas_grays_cerv_sym_correc_r2'; % PNG image
file_mask{2} = 'mask_grays_cerv_sym_correc_r2'; % PNG image
label_values{2} = [14 26 38 47 52 62 70 82 89 94 101 107 112 116 121 146 152 159 167 173 180 187 194 199 204 208 214 219 224 230];
z_slice_ref{2} = z_vertebrae(4); 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Start of the pipeline %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ext = '.nii.gz';
ext_atlas = '.png';
ext_aff = '.txt';
path_out = 'WMtracts_v10_results/';
path_ctrl = [path_out 'registered_template/'];
mkdir(path_out);
mkdir(path_ctrl);
prefix_out = 'WMtract_';

perc_up = 100*interp_factor;
perc_dn = 100/interp_factor;
prefix_ants = [path_out 'reg_'];
suffix_ants = '_reg';
coreg_iterations = 4;

template_cropped = [path_out file_template '_c6v'];
template_cropped_interp = [template_cropped '_int' num2str(interp_factor)];
template_cropped_intgauss = [template_cropped '_intgauss' num2str(interp_factor)];
templateci_thresh = [template_cropped_interp '_thrp6'];

min_ref_indx = 1;
numtracts = 100;
z_slice_ref_indx = 1:length(z_slice_ref);
z_disks_mid_noref = z_disks_mid;

templateci_slice_ref = cell(number_data_slices,1);
templatecig_slice_ref = cell(number_data_slices,1);
templateci_slice_ref_thresh = cell(number_data_slices,1);
templatecit_slice_ref = cell(number_data_slices,1);

atlas_nifti = cell(number_data_slices,1);
mask_atlas = cell(number_data_slices,1);
num_slice_ref = cell(number_data_slices,1);
suffix_ants_ref = cell(number_data_slices,1);

for ref_indx = 1:number_data_slices
    
    numtracts = min(numtracts,length(label_values{ref_indx}));
    if (length(label_values{ref_indx}) < length(label_values{min_ref_indx}))
        min_ref_indx = ref_indx; 
    end
    
    z_slice_ref_indx(ref_indx) = z_slice_ref{ref_indx};
    ind = find(z_disks_mid_noref == z_slice_ref{ref_indx});
    z_disks_mid_noref = [z_disks_mid_noref(1:ind-1) z_disks_mid_noref(ind+1:end)];
    
    templateci_slice_ref{ref_indx} = [template_cropped_interp '_slice_ref_' num2str(z_slice_ref{ref_indx})];
    templatecig_slice_ref{ref_indx} = [template_cropped_intgauss '_slice_ref_' num2str(z_slice_ref{ref_indx})];
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

reg_mask = cell(number_data_slices,numtracts/2);

for ref_indx = 1:number_data_slices
    for label = 1:numtracts/2
        reg_mask{ref_indx,label} = [path_out file_atlas{ref_indx} '_regmask' num2str(label)];
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Preliminary operations: cropping and interpolation of the template %%%

% Cropping the template
cmd = ['c3d ' file_template ext ' -trim 6vox -o ' template_cropped ext];
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
    num2str(perc_up) 'x' num2str(perc_up) 'x100% -o ' template_cropped_intgauss ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Binarization of the template for slice coregistration
cmd = ['c3d ' template_cropped_interp ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_thresh ext];
disp(cmd)
[status,result] = unix(cmd);
if(status), error(result); end

% Get the template reference slice associated with each atlas data
for ref_indx = 1:number_data_slices
    
    cmd = ['c3d ' template_cropped_interp ext ' -slice z ' num2str(z_slice_ref{ref_indx}) ' -o ' templateci_slice_ref{ref_indx} ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    cmd = ['c3d ' template_cropped_intgauss ext ' -slice z ' num2str(z_slice_ref{ref_indx}) ' -o ' templatecig_slice_ref{ref_indx} ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    cmd = ['c3d ' templateci_slice_ref{ref_indx} ext ' -pim r -threshold 0% 60% 0 1 -o ' templateci_slice_ref_thresh{ref_indx} ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
end

% Save each atlas and mask into a nifti with the same scales as the template
% One separate file for each tract
atlas_min = imread([file_atlas{min_ref_indx} ext_atlas]);

for ref_indx = 1:number_data_slices
    
    [~,~,scales] = read_avw(templateci_slice_ref{ref_indx});
    atlas = imread([file_atlas{ref_indx} ext_atlas]);
    mask = imread([file_mask{ref_indx} ext_atlas]);
    mask = m_normalize(mask);
    
    save_avw(mask,mask_atlas{ref_indx},'d',scales);
    
    % Select only tracts present in all references
    values = sort([0 4 255 label_values{ref_indx}]);
    atlas = m_quantify_image(atlas,values);
    atlas_partial_ML = 255 * ones(size(atlas));
    atlas_partial_ML (atlas == 0) = 0;
    labels_slice = unique(atlas_min);
    labels_slice = labels_slice(:)';
    labels_slice = labels_slice(labels_slice > 0);
    labels_slice = labels_slice(labels_slice < 255);
    for label = labels_slice
        atlas_partial_ML(atlas == label) = label;
    end
    atlas_partial_ML = m_clean_points(atlas_partial_ML,255);
    save_avw(atlas_partial_ML,atlas_nifti{ref_indx},'d',scales);
    
    % Creating masks for registration
    % One mask for each left-right couple of tracts
    for label = 1:numtracts/2
        temp = 255 * ones(size(atlas_partial_ML));
        temp(atlas_partial_ML == 0) = 0;
        temp(atlas_partial_ML == 4) = 4;
        temp( atlas_partial_ML == label_values{min_ref_indx}(label) ) = label_values{min_ref_indx}(label);
        temp( atlas_partial_ML == label_values{min_ref_indx}(numtracts/2+label) ) = label_values{min_ref_indx}(numtracts/2+label);
        
        save_avw(temp,reg_mask{ref_indx,label},'d',scales);
    end
    
    
    for label = 1:numtracts
        temp = zeros(size(atlas_partial_ML));
        ind = find( atlas_partial_ML == label_values{min_ref_indx}(label) );
        temp(ind) = 1;
        
        tract_atlas = [path_out 'tract_atlas' num2str(z_slice_ref{ref_indx}) '_' num2str(label)];
        save_avw(temp,tract_atlas,'d',scales);
    end
    
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

clear templateCI templateCROP;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Registration of atlas data into template space %%%%%%%%%%%%%


for ref_indx = 1:number_data_slices
    
    
    % Registration transform computation
    cmd = ['ants 2 -o ' prefix_ants ' ',...
        '-m PSE[' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{ref_indx} ext ',' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{ref_indx} ext ',0.5,100,11,0,10,1000] ',...
        '-m MSQ[' templateci_slice_ref_thresh{ref_indx} ext ',' mask_atlas{ref_indx} ext ',0.5,0] ',...
        '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
        '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    % Constraint the warping field to preserve symmetry
    Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
    
    % Applying tranform to the mask
    cmd = ['WarpImageMultiTransform 2 ' mask_atlas{ref_indx} ext ' ' mask_atlas{ref_indx} suffix_ants_ref{ref_indx} ext ' ',...
        Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    % Applying tranform to the initial atlas
    cmd = ['WarpImageMultiTransform 2 ' atlas_nifti{ref_indx} ext ' ' atlas_nifti{ref_indx} suffix_ants_ref{ref_indx} ext ' ',...
        Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status),error(result);end
    
    
    % Applying tranform to the tract files and copying geometry and saving
    for label = 1:numtracts/2
        tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{ref_indx}) '_' num2str(label)];
        tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{ref_indx}) '_' num2str(label+numtracts/2)];
        
        cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants_ref{ref_indx} ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants_ref{ref_indx} ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        cmd = ['WarpImageMultiTransform 2 ' reg_mask{ref_indx,label} ext ' ' reg_mask{ref_indx,label} suffix_ants_ref{ref_indx} ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templateci_slice_ref_thresh{ref_indx} ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        cmd = ['c3d ' templateci_slice_ref{ref_indx} ext ' ' tract_atlas_g suffix_ants_ref{ref_indx} ext ' -copy-transform -o ' tract_atlas_g suffix_ants_ref{ref_indx} ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        cmd = ['c3d ' templateci_slice_ref{ref_indx} ext ' ' tract_atlas_d suffix_ants_ref{ref_indx} ext ' -copy-transform -o ' tract_atlas_d suffix_ants_ref{ref_indx} ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        cmd = ['c3d ' templateci_slice_ref{ref_indx} ext ' ' reg_mask{ref_indx,label} suffix_ants_ref{ref_indx} ext ' -copy-transform -o ' reg_mask{ref_indx,label} suffix_ants_ref{ref_indx} ext];
        disp(cmd)
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
        tractsHR{label+numtracts/2}(:,:,num_slice_ref{ref_indx}) = temp_d;
    end
    
    % Get slice from the thresholded template for later registration
    cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(z_slice_ref{ref_indx}) ' -o ' templatecit_slice_ref{ref_indx} ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% Coregistration for intermediate vertebral levels %%%%%%%%%%%%


num_iter = 0;



for zslice = z_disks_mid_noref
    
    num_iter = num_iter + 1;
    num_slice = zslice + 1;
    
    templatecit_slice = [templateci_thresh '_slice' num2str(zslice)];
    templatecig_slice = [template_cropped_intgauss '_slice' num2str(zslice)];
    
    cmd = ['c3d ' templateci_thresh ext ' -slice z ' num2str(zslice) ' -o ' templatecit_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    cmd = ['c3d ' template_cropped_intgauss ext ' -slice z ' num2str(zslice) ' -o ' templatecig_slice ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
    
    % ------------------------------ CASE 1 -------------------------------
    if (zslice > z_slice_ref{1})
        % Above first reference -- Only first ref
        
        
        sliceref_thresh_bottom = templatecit_slice_ref{1};
        atlas_slice = [atlas_nifti{1} suffix_ants_ref{1}];
        atlas_slice_reg = [atlas_slice suffix_ants num2str(zslice)];
        
        % Registration transform computation
        cmd = ['ants 2 -o ' prefix_ants ' ',...
            '-m PSE[' templatecit_slice ext ',' sliceref_thresh_bottom ext ',' templatecit_slice ext ',' sliceref_thresh_bottom ext ',0.5,100,11,0,10,1000] ',...
            '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_bottom ext ',0.5,0] ',...
            '-m MI[' templatecig_slice ',' templatecig_slice ',0.4,32] ',...
            '--use-all-metrics-for-convergence 1 ',...
            '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
            '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
        disp(cmd)
        [statusANTS,resultANTS] = unix(cmd);
        if(statusANTS), error(resultANTS); end
        
        % Constraint the warping field to preserve symmetry
        Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
        
        % Apply transform to the initial atlas as a control
        cmd = ['WarpImageMultiTransform 2 ' atlas_slice ext ' ' atlas_slice_reg ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        
        % Apply tranform to the tract files and constraint to be symmetric
        for label = 1:numtracts/2
            
            tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{1}) '_' num2str(label) suffix_ants_ref{1}];
            tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{1}) '_' num2str(label+numtracts/2) suffix_ants_ref{1}];
            
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            disp(cmd)
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            disp(cmd)
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
            disp(cmd)
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
            disp(cmd)
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
            tractsHR{label+numtracts/2}(:,:,num_slice) = temp_d;
            
        end
        
        
        % Move control files to control folder
        [scs,msg] = movefile([atlas_slice_reg ext],path_ctrl);
        if(~scs), error(msg); end
        
        % Remove auxiliary files
        cmd = ['rm ' templatecit_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status), error(result); end
        
        %------------------------------------------------------------------
        
        
        
        
        % ---------------------------- CASE 2 -----------------------------
    elseif (zslice < z_slice_ref{end})
        % Below last reference -- Only last ref 
        
        
        sliceref_thresh_top = templatecit_slice_ref{end};
        atlas_slice = [atlas_nifti{end} suffix_ants_ref{end}];
        atlas_slice_reg = [atlas_slice suffix_ants num2str(zslice)];
        
        % Registration transform computation
        cmd = ['ants 2 -o ' prefix_ants ' ',...
            '-m PSE[' templatecit_slice ext ',' sliceref_thresh_top ext ',' templatecit_slice ext ',' sliceref_thresh_top ext ',0.5,100,11,0,10,1000] ',...
            '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_top ext ',0.5,0] ',...
            '-m MI[' templatecig_slice ',' templatecig_slice ',0.4,32] ',...
            '--use-all-metrics-for-convergence 1 ',...
            '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
            '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
        disp(cmd)
        [statusANTS,resultANTS] = unix(cmd);
        if(statusANTS), error(resultANTS); end
        
        % Constraint the warping field to preserve symmetry
        Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
        
        % Apply transform to the initial atlas as a control
        cmd = ['WarpImageMultiTransform 2 ' atlas_slice ext ' ' atlas_slice_reg ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        % Apply tranform to the tract files and constraint to be symmetric
        for label = 1:numtracts/2
            
            tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{end}) '_' num2str(label) suffix_ants_ref{end}];
            tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{end}) '_' num2str(label+numtracts/2) suffix_ants_ref{end}];
            
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            disp(cmd)
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ',...
                Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
            disp(cmd)
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
            disp(cmd)
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
            disp(cmd)
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
            tractsHR{label+numtracts/2}(:,:,num_slice) = temp_d;
            
        end
        
        
        % Move control files to control folder
        [scs,msg] = movefile([atlas_slice_reg ext],path_ctrl);
        if(~scs), error(msg); end
        
        % Remove auxiliary files
        cmd = ['rm ' templatecit_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status), error(result); end
        
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
        
        diff_top = abs(z_slice_ref{indx_ref_top} - zslice);
        diff_bottom = abs(zslice - z_slice_ref{indx_ref_bottom});
        
        if (diff_top < diff_bottom)
            sliceref_thresh_far = sliceref_thresh_bottom;
            sliceref_thresh_close = sliceref_thresh_top;
            indx_ref_close = indx_ref_top;
            indx_ref_far = indx_ref_bottom;
        else
            sliceref_thresh_far = sliceref_thresh_top;
            sliceref_thresh_close = sliceref_thresh_bottom;
            indx_ref_close = indx_ref_bottom;
            indx_ref_far = indx_ref_top;
        end
        
        weight_close = abs(z_slice_ref{indx_ref_far} - zslice) / abs(z_slice_ref{indx_ref_close} - z_slice_ref{indx_ref_far});
        
        atlas_slice_close = [atlas_nifti{indx_ref_close} suffix_ants_ref{indx_ref_close}];
        atlas_slice_close_reg = [atlas_slice_close suffix_ants num2str(zslice)];
        atlas_slice_close_ctrl = [atlas_slice_close_reg '_ctrl'];
        atlas_slice_far = [atlas_nifti{indx_ref_far} suffix_ants_ref{indx_ref_far}];
        atlas_slice_far_reg = [atlas_slice_far suffix_ants num2str(zslice)];
        warp_shape = [prefix_ants 'warp_shape'];
        affine_shape = [prefix_ants 'affine_shape'];
        
        % Registration transform computation for further ref slice
        cmd = ['ants 2 -o ' prefix_ants ' ',...
            '-m PSE[' templatecit_slice ext ',' sliceref_thresh_far ext ',' templatecit_slice ext ',' sliceref_thresh_far ext ',0.5,100,11,0,10,1000] ',...
            '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_far ext ',0.5,0] ',...
            '-m MI[' templatecig_slice ',' templatecig_slice ',0.4,32] ',...
            '--use-all-metrics-for-convergence 1 ',...
            '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
            '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
        disp(cmd)
        [statusANTS,resultANTS] = unix(cmd);
        if(statusANTS), error(resultANTS); end
        
        % Constraint the warping field to preserve symmetry
        Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
        
        % Apply transform to the further atlas
        cmd = ['WarpImageMultiTransform 2 ' atlas_slice_far ext ' ' atlas_slice_far_reg ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        % Registration transform computation for closer ref slice
        cmd = ['ants 2 -o ' prefix_ants ' ',...
            '-m PSE[' templatecit_slice ext ',' sliceref_thresh_close ext ',' templatecit_slice ext ',' sliceref_thresh_close ext ',0.5,100,11,0,10,1000] ',...
            '-m MSQ[' templatecit_slice ext ',' sliceref_thresh_close ext ',0.5,0] ',...
            '-m MI[' templatecig_slice ',' templatecig_slice ',0.4,32] ',...
            '--use-all-metrics-for-convergence 1 ',...
            '-t SyN[0.2] -r Gauss[3,0.] -i 1000x500x300 ',...
            '--rigid-affine true --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'];
        disp(cmd)
        [statusANTS,resultANTS] = unix(cmd);
        if(statusANTS), error(resultANTS); end
        
        % Constraint the warping field to preserve symmetry
        Warp_sym = m_symmetrize_2D_field([prefix_ants 'Warp'],ext);
        
        % Apply transform to the closer atlas
        cmd = ['WarpImageMultiTransform 2 ' atlas_slice_close ext ' ' atlas_slice_close_reg ext ' ',...
            Warp_sym ext ' ' prefix_ants 'Affine' ext_aff ' -R ' templatecit_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        % Rename shaping transforms
        [scs,msg] = movefile([prefix_ants 'Affine' ext_aff],[affine_shape ext_aff]);
        if(~scs), error(msg); end
        [scs,msg] = movefile([Warp_sym ext],[warp_shape ext]);
        if(~scs), error(msg); end
        
        
        % Apply tranform to the tract files and constraint to be symmetric
        for label = 1:numtracts/2
            
            reg_atlas_close = [reg_mask{indx_ref_close,label} suffix_ants_ref{indx_ref_close}];
            reg_atlas_far = [reg_mask{indx_ref_far,label} suffix_ants_ref{indx_ref_far}];
            
            % Coregistration between closer and further references
            [mean_reg,warps_close2ref,warps_far2ref] = m_coregister_multilabel(reg_atlas_close,reg_atlas_far,ext,weight_close,0,coreg_iterations,1);
            
            tract_atlas_g = [path_out 'tract_atlas' num2str(z_slice_ref{indx_ref_close}) '_' num2str(label) suffix_ants_ref{indx_ref_close}];
            tract_atlas_d = [path_out 'tract_atlas' num2str(z_slice_ref{indx_ref_close}) '_' num2str(label+numtracts/2) suffix_ants_ref{indx_ref_close}];
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_g ext ' ' tract_atlas_g suffix_ants ext ' ',...
                warps_close2ref{1} ext ' ' warps_close2ref{2} ext ' ' warps_close2ref{3} ext ' ' warps_close2ref{4} ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
            disp(cmd)
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['WarpImageMultiTransform 2 ' tract_atlas_d ext ' ' tract_atlas_d suffix_ants ext ' ',...
                warps_close2ref{1} ext ' ' warps_close2ref{2} ext ' ' warps_close2ref{3} ext ' ' warps_close2ref{4} ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
            disp(cmd)
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_g suffix_ants ext ' -copy-transform -o ' tract_atlas_g suffix_ants ext];
            disp(cmd)
            [status,result] = unix(cmd);
            if(status),error(result);end
            
            cmd = ['c3d ' templatecit_slice ext ' ' tract_atlas_d suffix_ants ext ' -copy-transform -o ' tract_atlas_d suffix_ants ext];
            disp(cmd)
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
            tractsHR{label+numtracts/2}(:,:,num_slice) = temp_d;
            
        end
        
        
        % Apply to closer ref as a control
        cmd = ['WarpImageMultiTransform 2 ' atlas_slice_close ext ' ' atlas_slice_close_ctrl ext ' ',...
            warps_close2ref{1} ext ' ' warps_close2ref{2} ext ' ' warps_close2ref{3} ext ' ' warps_close2ref{4} ext ' ' warp_shape ext ' ' affine_shape ext_aff ' -R ' templatecit_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status),error(result);end
        
        % Move control files to control folder
        [scs,msg] = movefile([atlas_slice_close_ctrl ext],path_ctrl);
        if(~scs), error(msg); end
        
        % Remove auxiliary files
        cmd = ['rm ' templatecit_slice ext];
        disp(cmd)
        [status,result] = unix(cmd);
        if(status), error(result); end
        
        
        %------------------------------------------------------------------
        
    end % if zslice position
    
    
    
end % for zslice







%%%%%%%%%%%%%%%%%% Interpolation between computed slices %%%%%%%%%%%%%%%%%%

for label = 1:numtracts
    for k = 1:length(z_disks_mid)-1
    
        tractsHR{label} = m_linear_interp(tractsHR{label},z_disks_mid(k)+1,z_disks_mid(k+1)+1);
%         tractsHR{label}(tractsHR{label} > 0.5) = 1; % ?
%         tractsHR{label}(tractsHR{label} < 0.5) = 0; % ?
        
    end
end




%%%%%%%%%%%%%%% Downsampling and partial volume computation %%%%%%%%%%%%%%%

for label = 1:numtracts
    for zslice = 0:513
        
        num_slice = zslice+1;
        
        tracts{label}(:,:,num_slice) = dnsamplelin(tractsHR{label}(:,:,num_slice),interp_factor);
        
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%% Writing the outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%

for label = 1:numtracts
    
    % Save ML version and copy geometry
    filetractML = [path_out prefix_out 'MLb_' num2str(label)];
    save_avw(tracts{label},filetractML,'d',scalesCROP);
    cmd = ['c3d ' template_cropped ext ' ' filetractML ext ' -copy-transform -o ' filetractML ext];
    disp(cmd)
    [status,result] = unix(cmd);
    if(status), error(result); end
    
end









