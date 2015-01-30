function [label] = sct_label(label)
% Compute a segmented spinal cord centerline 
%
% N.B. THE INPUT SHOULD BE .nii, not .nii.gz!!!
%
%The code library segments the spinal cord in the axial plane of a NIfTI image. 
%The output of the segmentation gives a parametrization of the cord surface, which 
%consists of the coordinates of the centerline and the radii for each angle.
%T1- and T2-weighted images are supported.
% 
%
% SYNTAX
% ========================================================================
% label=sct_label(label)
%
%
% INPUTS
% ========================================================================
% label
%   input_anat                   string. File name of input spine  T2 MR image. 
%   input_path                   string. path for the input file
%   (output_path)                string. path for output
%   (output_labeled_centerline)  string. File name of outpout labeled centerline of the spinal cord. 
%   (output_labeled_surface)     string. File name of outpout labeled surface of the spinal cord.
%   (surface_do)                 1 or 0 give as output a labeled surface
%                                too
%   (segmentation.do)            compute a segmentation of the spinal cord
%                                (if you don't have input centerline or surface MR image)
%   (segmentation)               structure of parameters for the
%                                segmentation of the spinal cord
%   (input_centerline)           string. File name of input centerline of the
%                                spinal cord
%   (input_surface)              string. File name of input surface of the
%                                spinal cord
%   (shift_AP)                   shift the centerline on the spine in mm default : 17 mm
%   (size_AP)                    mean around the centerline in the anterior-posterior direction in mm
%   (size_RL)                    mean around the centerline in the right-left direction in mm
%   (verbose)                    display figures
%
%
% OUTPUTS
% ========================================================================
% labeled_centerline
%(surface_centerline)
%
% DEPENDENCES
% ========================================================================
% - FSL       
% - SPM
%
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

if ~isfield(label.segmentation,'interval'), label.segmentation.interval=30; end
if ~isfield(label.segmentation,'nom_radius'), label.segmentation.nom_radius=5; end
if ~isfield(label.segmentation,'tolerance'), label.segmentation.tolerance=0.01; end
if ~isfield(label.segmentation,'ratio_criteria'), label.segmentation.ratio_criteria=0.05; end
if ~isfield(label.segmentation,'num_angles'), label.segmentation.num_angles=64; end
if ~isfield(label.segmentation,'update_multiplier'), label.segmentation.update_multiplier=0.8; end
if ~isfield(label.segmentation,'shear_force_multiplier'), label.segmentation.shear_force_multiplier=0.5; end
if ~isfield(label.segmentation,'max_coeff_horizontal'), label.segmentation.max_coeff_horizontal=10; end
if ~isfield(label.segmentation,'max_coeff_vertical'), label.segmentation.max_coeff_vertical=10; end

  

            j_disp(label.log,['\n\n   SPINAL CORD SEGMENTATION:'])
            j_disp(label.log,['-----------------------------------------------'])

            % Creates the scs_results folder
            warning off MATLAB:MKDIR:DirectoryExists            
            input{1,1} =  [label.input_path,label.input_anat,label.ext];
            output{1,1} = [label.output_path,'label_mat'];  % all matrices in .mat
            output{1,2} = [label.output_path,label.segmentation.centerline];                        % centerline in .nii
            output{1,3} = [label.output_path,label.segmentation.surface];
            output{1,4} = [label.output_path,label.segmentation.straightened];% segmented surface in .nii
  

            %% Parameters of the algorithm

            param.image_type             = label.segmentation.image_type;            % 1=T1; 2=T2
            param.interval               = label.segmentation.interval;             % Interval in mm between two slices for the initialization

            param.nom_radius             = label.segmentation.nom_radius;            % Nominal radius in mm that reprensents the initial estimate
            param.tolerance              = label.segmentation.tolerance;          % Percentage of the nominal radius that is used as the criterion to determine convergence
            param.ratio_criteria         = label.segmentation.ratio_criteria;     % Percentage of radius that must meet the tolerance factor to increment the coefficients

            param.num_angles             = label.segmentation.num_angles;           % Number of angles used
            param.update_multiplier      = label.segmentation.update_multiplier;   % Multiplies the force applied to deform the radius
            param.shear_force_multiplier = label.segmentation.shear_force_multiplier;  % Multiplies the shear force used to stay near the user defined center line. 
            param.max_coeff_horizontal   = label.segmentation.max_coeff_horizontal; % Maximal coefficient used to smooth the radius in the horizontal plane
            param.max_coeff_vertical     = label.segmentation.max_coeff_vertical;   % Maximal coefficient used to smooth the radius in the vertical direction (depends on the number of slices)

            param.ext   = label.ext;
            param.file_type = input{1,1}(end-2:end);
            param.nbIterMax = 30;            % Maximal number of iterations for each coefficients (if there is no convergence)
            param.resampling = 1;            % Resampling of the image
            param.language = 'English';      % Defines the language of the error display. Only 'English' and 'Français' are currently supported



            %% Execution of spinal cord segmentation

            if exist([input{1,1} '.nii']) == 2 || exist([input{1,1} '.nii.gz']) == 2 || exist(input{1,1}) == 2
                %try
                    spinal_cord_segmentation_horsfield(input{1,1},output{1,1},param);

            else
                error(['File ' input{1,1} ' doesn''t exist'])
            end
            
                        %% Straigthening of the image
            scs_straightening(output{1,1},output{1,4});
            %% Saving the centerline and the segmented cord surface in NIfTI
            scs_nifti_save_v4(output{1,1},output{1,2},output{1,3},output{1,4},param);

            j_disp(label.log,['... File created: ',output{1,2},output{1,3}])
end