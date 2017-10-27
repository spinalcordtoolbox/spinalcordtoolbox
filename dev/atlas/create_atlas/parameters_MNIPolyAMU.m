% Parameters for creating WM atlas for template: MNI-Poly-AMU
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
path_template = ['/Users/julien/data/sct_dev/MNI-Poly-AMU/template/'];
% name of the WM template to build the atlas from. Don't put the extension.
file_template = 'MNI-Poly-AMU_WM';
which_template = 'MNI-Poly-AMU_WM';
% slice number corresponding to the mid-C4 level (where the atlas is derived from)
z_slice_ref = 387;
% interpolation factor for the MNI-Poly-AMU template in order to match the hi-res grays atlas
interp_factor = 6;
% these are the value corresponding to the slice number (z) on the MNI-Poly-AMU template, at which the atlas will be warped. It corresponds to the mid-levels as well as the level of the intervertebral disks.
% NB: to extract these values, you have to look at the T2 and WM template, because this script will crop the WM template (which can be smaller than the T2), therefore the maximum z cannot exceed the zmax that will be generated in the cropped version of the WM template.
z_disks_mid = [483 476 466 455 440 423 406 387 371 356 339 324 303 286 268 248 229 208 186 166 143 122 98 79 53 35 13 0];
% same as before-- except that C4 mid-vertebral is not listed. 
z_disks_mid_noC4 = [483 476 466 455 440 423 406 371 356 339 324 303 286 268 248 229 208 186 166 143 122 98 79 53 35 13 0];

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
prefix_out = 'WMtract__';
