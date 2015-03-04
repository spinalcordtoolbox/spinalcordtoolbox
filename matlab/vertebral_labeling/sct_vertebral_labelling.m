function sct_vertebral_labelling(nifti, centerline, varargin)

% BATCH_LABELING 
%
% DETAILED DESCRIPTION.
% Use this script to run vertebrare_labeling_T1.m and
% vertebrae_labeling_T2.m
%
% SYNTAX
% ========================================================================
% batch_labeling
%
%
%
% COMMENTS
% ========================================================================  
% Use this script to run vertebrare_labeling_T1.m and
% vertebrae_labeling_T2.m
%
% Copyright (c) 2013  NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
% Created by: Eugenie Ullmann
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


dbstop if error

p=inputParser;

addRequired(p,'nifti',@isstr);
addRequired(p,'centerline',@isstr);
addOptional(p,'contrast', 'T1', @(x) any(validatestring(x,{'T1', 'T2'})));

parse(p,nifti,centerline, varargin{:})

contrast=p.Results.contrast;

 
% PATH AND FILE NAME FOR ANATOMICAL IMAGE
[label.input_anat, label.input_path, label.ext] = sct_tool_remove_extension(p.Results.nifti, 0);

% PATH FOR OUTPUT
label.output_path=label.input_anat;
label.output_labeled_centerline=[contrast '_centerline'];
label.output_labeled_surface=[contrast '_surface']; % optional


% =======================================================
% OTHER PARAMETERS
% =======================================================
label.surface_do=0;

% =======================================================
% Spinal Cord Segmentation Parameters 

label.segmentation.do=0;
label.segmentation.interval = 30 ;             % Interval in mm between two slices for the initialization

label.segmentation.nom_radius= 5;            % Nominal radius in mm that reprensents the initial estimate
label.segmentation.tolerance = 0.01;          % Percentage of the nominal radius that is used as the criterion to determine convergence
label.segmentation.ratio_criteria = 0.05;     % Percentage of radius that must meet the tolerance factor to increment the coefficients

label.segmentation.num_angles = 64;           % Number of angles used
label.segmentation.update_multiplier = 0.8;   % Multiplies the force applied to deform the radius
label.segmentation.shear_force_multiplier= 0.5;  % Multiplies the shear force used to stay near the user defined center line. 
label.segmentation.max_coeff_horizontal = 10; % Maximal coefficient used to smooth the radius in the horizontal plane
label.segmentation.max_coeff_vertical = 10;   % Maximal coefficient used to smooth the radius in the vertical direction (depends on the number of slices)
label.segmentation.centerline='centerline';
label.segmentation.surface='surface';
label.segmentation.straightened='straightened';
label.log = 'log_segmentation';

% OR

% Spinal Cord labeling Parameters 
label.input_centerline=p.Results.centerline; % optional
label.input_surface='segmentation_binary'; % optional  
% =======================================================


label.shift_AP=17; % shift the centerline on the spine in mm default : 17 mm
label.size_AP=6; % mean around the centerline in the anterior-posterior direction in mm
label.size_RL=5; % mean around the centerline in the right-left direction in mm

label.verbose       = 1; % display figures


%=========================================================================

if label.segmentation.do
    
    j_disp(label.log,['\n\n\n=========================================================================================================='])
    j_disp(label.log,['   Spinal Cord Segmentation'])
    j_disp(label.log,['=========================================================================================================='])
    
    % process segentation
    if strcmp(contrast,'T1')
        label.segmentation.image_type = 1;
    elseif strcmp(contrast,'T2')
        label.segmentation.image_type = 2;
    end
    sct_label(label);
    
end

% labeling process 
if strcmp(contrast,'T1')
    labeling_vertebrae_T1(label);
elseif strcmp(contrast,'T2')
    labeling_vertebrae_T2(label)
end


