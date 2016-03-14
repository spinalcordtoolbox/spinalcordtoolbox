function sct_seg(nifti)
% Compute a segmented spinal cord centerline 
% 
%
% SYNTAX
% ========================================================================
% label=sct_label(label)

% DEPENDENCES
% ========================================================================
% - FSL       
%
% Copyright (c) 2013  NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
% Created by: Eugénie Ullmann
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.
% ========================================================================

% TODO: enable to input manual labeling (to play with parameters and be able to reproduce)
% TODO: by default, select all slices
% TODO: zoom window for initialization
% TODO: active contour on Laplacian (indep. of T1 or T2).

[fname,path,ext]=sct_tool_remove_extension(nifti,1);

            disp(['\n\n   SPINAL CORD SEGMENTATION:'])
            disp(['-----------------------------------------------'])

            % Creates the scs_results folder
            warning off MATLAB:MKDIR:DirectoryExists            
            input{1,1} =  nifti;
            output{1,1} = [path,'label_mat'];  % all matrices in .mat
            output{1,2} = [fname '_centerline']; % centerline in .nii
            output{1,3} = [fname '_surface']; % segmentation
            output{1,4} = [fname '_straight'];% straight spinal cord
  

            %% Parameters of the algorithm

            param.image_type             = 2;    % 1=T1; 2=T2
            param.interval               = 30;   % Interval in mm between two slices for the manual initialization

            param.nom_radius             = 2;    % Nominal radius in mm that reprensents the initial estimate. Default=5
            param.tolerance              = 0.01; % Percentage of the nominal radius that is used as the criterion to determine convergence. Default=0.01
            param.ratio_criteria         = 0.05; % Percentage of radius that must meet the tolerance factor to increment the coefficients. Default=0.05

            param.num_angles             = 64;   % Number of angles used
            param.update_multiplier      = 0.8;  % Multiplies the force applied to deform the radius
            param.shear_force_multiplier = 0.5;  % Multiplies the shear force used to stay near the user defined center line. Default=0.5
            param.max_coeff_horizontal   = 10;   % Maximal coefficient used to smooth the radius in the horizontal plane
            param.max_coeff_vertical     = 10;   % Maximal coefficient used to smooth the radius in the vertical direction (depends on the number of slices)

            param.ext   = ext;
            param.file_type = input{1,1}(end-2:end);
            param.nbIterMax = 30;                % Maximal number of iterations for each coefficients (if there is no convergence)
            param.resampling = 1;                % Resampling of the image
            param.language = 'English';          % Defines the language of the error display. Only 'English' and 'Français' are currently supported


            %% Execution of spinal cord segmentation
            if exist([input{1,1} '.nii']) == 2 || exist([input{1,1} '.nii.gz']) == 2 || exist(input{1,1}) == 2
                %try
                    spinal_cord_segmentation_horsfield(input{1,1},output{1,1},param);
            else
                error(['File ' input{1,1} ' doesn''t exist'])
            end
            
            %% Straigthening of the image
            scs_straightening(output{1,1}, output{1,4});

            %% Saving the centerline and the segmented cord surface in NIfTI
            scs_nifti_save_v4(output{1,1}, output{1,2}, output{1,3}, output{1,4}, param);

            disp(['... Done! To see results type:'])
            % TODO: reorient surface
            disp(['fslview t2.nii.gz ',output{1,3},' -l Red -b 0,1 -t 0.7 &'])
end