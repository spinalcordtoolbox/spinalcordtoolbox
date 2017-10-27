% Parameters for creating WM atlas for template: PAM50
% Author: Julien Cohen-Adad, 2017

%--------------------------------------------------------------------------
% LABEL PARAMETERS
%--------------------------------------------------------------------------
% full path of the images that contains the raw atlas
path_atlas_data = ['/Users/julien/temp/raw_data/'];
% file name of the 2D atlas (no extension)
file_atlas = 'atlas_grays_cerv_sym_correc_r6';
% file name of the 2D binary WM mask associated with the atlas, which will be used for registration to the MRI template (no extension)
file_mask = 'mask_grays_cerv_sym_correc_r5';
% extensions of the images
ext_atlas = '.png';
% CSV text file that contains label ID, values and description
file_atlas_txt = 'atlas_grays_cerv_sym_correc_r6_label.txt';

%--------------------------------------------------------------------------
% TEMPLATE PARAMETERS
%--------------------------------------------------------------------------
% define path of template
path_template = ['/Users/julien/data/sct_dev/PAM50/template/'];
% name of the WM template to build the atlas from. Don't put the extension.
file_template = 'PAM50_wm';
which_template = 'PAM50';
% slice number corresponding to the mid-C4 level (where the atlas is derived from)
z_slice_ref = 850;
% interpolation factor for the template in order to match the hi-res grays atlas
interp_factor = 6;
% size to crop the template (only along X and Y) for computational efficiency
crop_size = '43x33x1100vox'; 
% z-values at which the atlas will be warped. It is a compromise between accuracy (small z-step) and processing time (large z-step). The atlas will then be interpolated for the intermediate slices.
z_disks_mid = [80:5:990];  %[[80:5:990], 990];

%--------------------------------------------------------------------------
% OUTPUT PARAMETERS
%--------------------------------------------------------------------------
% path of output results (add "/" at the end)
path_out = ['./atlas_', date, filesep];
% temp folder for WMtracts
folder_tracts = ['WMtracts_outputs/'];
% temp folder for registered template
folder_ctrl = ['registered_template/'];
% final output folder
folder_final = ['final_results/'];
% prefix of output files
prefix_out = 'PAM50_atlas';
