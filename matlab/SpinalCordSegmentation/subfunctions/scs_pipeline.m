%% Scs_pipeline
% This script is an exemple of how the section using the
% spinal_cord_segmentation algorithm could look like. After declaring the
% input and output of the functions, the parameters of 
% spinal_cord_segmentation are declared. It is possible to suppress certain
% parameter as they are given default value by the function. 
% The script then executes spiual_cord_segmentation for all the given input.
% There are 3 others function in the pipeline:
% Scs_display: displays the results for an output file of the scs algorithm
% Scs_straigthening: Straightens the volume of the image along the
% center-line. Results are shown with scs_slider_3dmatrix
% Scs_nifti_save: Saves the centerline and spinal cord surface in nifti
% format for subsequent image registration

%% Files Input/Output
clear variables

% path 
%path_name_in = '/Users/gbm4900/Downloads/';
%path_name_in = '/Users/gbm4900/code/';
%path_name_in = '/Users/Shared/template_mcgill/43/space/';
path_name_in = '';
no_patient = '43_';
path_name_out = './';

% Creates the scs_results folder
warning off MATLAB:MKDIR:DirectoryExists
mkdir(path_name_out,'scs_results')
%mkdir(path_name_out,'scs_results')
path_name_out = [path_name_out 'scs_results' filesep];
%path_name_out = '/Users/gbm4900/data/template_mcgill/43/space/';
input{1,1} = [path_name_in 'anat_data'];
%input{1,1} = [path_name_in 'space'];
output{1,1} = [path_name_out strrep(input{1,1},path_name_in,no_patient) datestr(now, '_yyyy_mm_dd')];   % all matrices in .mat

%input{1,2} = [path_name_in 'centerline.mat'];              % File containing the Center line coordinates for the automatic initialization
output{1,2} = [path_name_out strrep(input{1,1},path_name_in,no_patient) '_centerline'];                        % centerline in .nii
output{1,3} = [path_name_out strrep(input{1,1},path_name_in,no_patient) '_surface'];                           % segmented surface in .nii
output{1,4} = [path_name_out strrep(input{1,1},path_name_in,no_patient) '_straightened'];

%% Parameters of the algorithm

for i=1:size(input,1)
    param{i}.image_type = 1;            % 1=T1; 2=T2
    param{i}.interval = 25;             % Interval in mm between two slices for the initialization
 
    param{i}.nom_radius = 5;            % Nominal radius in mm that reprensents the initial estimate
    param{i}.tolerance = 0.01;          % Percentage of the nominal radius that is used as the criterion to determine convergence
    param{i}.ratio_criteria = 0.05;     % Percentage of radius that must meet the tolerance factor to increment the coefficients
    
    param{i}.num_angles = 64;           % Number of angles used
    param{i}.update_multiplier = 0.8;   % Multiplies the force applied to deform the radius
    param{i}.shear_force_multiplier = 0.5;  % Multiplies the shear force used to stay near the user defined center line. 
    param{i}.max_coeff_horizontal = 10; % Maximal coefficient used to smooth the radius in the horizontal plane
    param{i}.max_coeff_vertical = 10;   % Maximal coefficient used to smooth the radius in the vertical direction (depends on the number of slices)
    
    if size(input,2)>1
        if exist([input{i,2} '.mat']) == 2 % Check the existance of the output file to use the centerline from previous iterations
            load (input{i,2}, 'centerline') 
            param{i}.centerline=squeeze(centerline(end,:,:));
            clear centerline
        end
    end
    param{i}.file_type = input{i,1}(end-3:end);   % Gets the file extension from the input
    
    param{i}.nbIterMax = 5;            % Maximal number of iterations for each coefficients (if there is no convergence)
    param{i}.resampling = 1;            % Resampling of the image
    param{i}.language = 'English';      % Defines the language of the error display. Only 'English' and 'Français' are currently supported
   
end

%% Execution of spinal cord segmentation

for i=1:size(input,1)
    if exist([input{i,1} '.nii']) == 2 || exist([input{i,1} '.nii.gz']) == 2 || exist(input{i,1}) == 2
        try
            spinal_cord_segmentation_horsfield(input{i,1},output{i,1},param{i});
        catch exception
            disp(['Spinal cord segmentation failed for ' input{i,1} '. Reason: ' exception.message])
        end
    else
        errordlg(['File ' input{i,1} ' doesn''t exist'])
    end
end

%% Display of the results
close all
load(output{1,1})
scs_display(m_nifti, centerline,radius, slices(1), slices(2), angles, resampling, area_per_slice, average_area);

%% Straigthening of the image
scs_straightening(output{1,1},output{1,4});
load(output{1,4});
m_straight = permute(m_straight, [3 2 1]);
scs_slider_3dmatrix(m_straight, 'Visualization of the spinal cord straightened');

%% Saving the centerline and the segmented cord surface in NIfTI
scs_nifti_save(input{1,1},output{1,1},output{1,2},output{1,3},param);

