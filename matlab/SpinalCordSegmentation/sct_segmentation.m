 
function [sct] = sct_segmentation(sct)  

            j_disp(sct.log,['\n\n   SPINAL CORD SEGMENTATION:'])
            j_disp(sct.log,['-----------------------------------------------'])

            % Creates the scs_results folder
            warning off MATLAB:MKDIR:DirectoryExists
            mkdir([sct.segmentation.path_name_in],'Spinal_Cord_Segmentation')
            input{1,1} =  [sct.segmentation.path_name_in,'anat_data'];
            output{1,1} = [sct.segmentation.path_name_out,'scs_anat'];   % all matrices in .mat
            output{1,2} = [sct.segmentation.path_name_out,sct.segmentation.centerline];                        % centerline in .nii
            output{1,3} = [sct.segmentation.path_name_out,sct.segmentation.surface];                           % segmented surface in .nii
            output{1,4} = [sct.segmentation.path_name_out,'scs_anat_straightened'];

            %% Parameters of the algorithm

            param{1}.image_type             = sct.segmentation.image_type;            % 1=T1; 2=T2
            param{1}.interval               = sct.segmentation.interval;             % Interval in mm between two slices for the initialization

            param{1}.nom_radius             = sct.segmentation.nom_radius;            % Nominal radius in mm that reprensents the initial estimate
            param{1}.tolerance              = sct.segmentation.tolerance;          % Percentage of the nominal radius that is used as the criterion to determine convergence
            param{1}.ratio_criteria         = sct.segmentation.ratio_criteria;     % Percentage of radius that must meet the tolerance factor to increment the coefficients

            param{1}.num_angles             = sct.segmentation.num_angles;           % Number of angles used
            param{1}.update_multiplier      = sct.segmentation.update_multiplier;   % Multiplies the force applied to deform the radius
            param{1}.shear_force_multiplier = sct.segmentation.shear_force_multiplier;  % Multiplies the shear force used to stay near the user defined center line. 
            param{1}.max_coeff_horizontal   = sct.segmentation.max_coeff_horizontal; % Maximal coefficient used to smooth the radius in the horizontal plane
            param{1}.max_coeff_vertical     = sct.segmentation.max_coeff_vertical;   % Maximal coefficient used to smooth the radius in the vertical direction (depends on the number of slices)

            param{1}.file_type = input{1,1}(end-2:end);
            param{1}.nbIterMax = 30;            % Maximal number of iterations for each coefficients (if there is no convergence)
            param{1}.resampling = 1;            % Resampling of the image
            param{1}.language = 'English';      % Defines the language of the error display. Only 'English' and 'Français' are currently supported



            %% Execution of spinal cord segmentation

            if exist([input{1,1} '.nii']) == 2 || exist([input{1,1} '.nii.gz']) == 2 || exist(input{1,1}) == 2
                try
                    spinal_cord_segmentation_horsfield(input{1,1},output{1,1},param{1});
                catch exception
                    j_disp(sct.log,['spinal_cord_segmentation_horsfield(', input{1,1},',',output{1,1},',param)']); error(['Spinal cord segmentation failed for ' input{1,1} '. Reason: ' exception.message])
                end
            else
                error(['File ' input{1,1} ' doesn''t exist'])
            end



            %% Straigthening of the image
            scs_straightening(output{1,1},output{1,4});
            
            %% Saving the centerline and the segmented cord surface in NIfTI
            scs_nifti_save_v4(input{1,1},output{1,1},output{1,2},output{1,3},output{1,4},param);

            j_disp(sct.log,['... File created: ',output{1,2},output{1,3},output{1,4}])
end