function [] = spinal_cord_segmentation_horsfield(input,output,param)
% spinal_cord_segmentation
%   This script will compute the cross-sectional area of the spinal_cord of
%   the T1 or T2 weighted MRI image given by the input variable (Nifty or
%   Matlab format) using a method inspired by Horsfield et al.(2010). 
%   It will store the center-line, the radius for each angle
%   and each slice, the area of each slice and the average area of the cord
%   in the output file. The parameters are use to fine-tune the segmentation.
%
% SYNTAX:
% [] = scs_smoothing(INPUT,OUTPUT,PARAM)
%
% _________________________________________________________________________
% INPUTS:  
%
% INPUT
%   path and name of the file that contains the .mat and .nii MRI image.
%
% OUTPUT
%   path and name of the file in which the function will save the result of
%   the segmentation and computation of the cross-sectional area
% 
% PARAM
%   Contains different parameters that are used to fine-tune the
%   segmentation (see user guide for details)
% _________________________________________________________________________
% OUTPUTS:
%
% NONE
%
%% Initialization

global fname_log
fname_log = scs_log();

% Verification of the inputs and definition of the default parameters
param = scs_verif_param(param);

%  profile on % to start the profiler analysis

[m_nifti, m_volume, dims, scales, nom_radius, m_center_line, m_tangent_cord, param] = scs_initialization_v3(input, param);

%% Definition of certain variables for the iterative segmentation

m_tangent_cord_user=m_tangent_cord;
m_center_line_user=m_center_line;

% Definition of the angles used in the scs_radius_update function
angles_start = -pi();
angles_step = 2*pi()/(param.num_angles);
angles_end = pi()-pi()/(param.num_angles);
angles = angles_start : angles_step : angles_end;

% transformation in cartesian coordinates
coord_radius=[cos(angles); sin(angles); zeros(1,param.num_angles)];

% Initialization of the parameters for the convergence loop
criteria = 10;
nbIter=0;

% Initialisation of the radius values
m_previous_radius = nom_radius*ones(size(m_center_line,2),param.num_angles); 

% definition of the tolerance
tolerance = param.tolerance*nom_radius;

% definition of the step buffer
% step=(size(m_volume_sobel,3)-size(m_center_line,2));
step=0;

% Display of initial parameters
j_disp(fname_log,['\n\n\nFile parameters:']);
j_disp(fname_log,['.. Input Directory/File name:\t',input]);
j_disp(fname_log,['.. Output Directory/File name:\t',output]);
j_disp(fname_log,['\n\nImage parameters:']);
j_disp(fname_log,['.. Working plane:\t','Axial']);
j_disp(fname_log,['.. Image Type :\t\tT',num2str(param.image_type)]);
j_disp(fname_log,['.. File Type :\t\t',num2str(param.file_type)]);
j_disp(fname_log,['.. Dimensions (x,y,z):\t',num2str(dims)]);
j_disp(fname_log,['.. Scales (x,y,z,t):\t',num2str(scales)]);
j_disp(fname_log,['.. Start image:\t\t',num2str(param.slices(1))]);
j_disp(fname_log,['.. End image:\t\t',num2str(param.slices(2))]);
j_disp(fname_log,['.. Resampling:\t\t',num2str(param.resampling)]);
j_disp(fname_log,['\n\nAlgorithm parameters:']);
j_disp(fname_log,['.. Number of angles:\t\t\t',num2str(param.num_angles)]);
j_disp(fname_log,['.. Update multiplier:\t\t\t',num2str(param.update_multiplier)]);
j_disp(fname_log,['.. Nominal radius:\t\t\t',num2str(param.nom_radius)]);
j_disp(fname_log,['.. Maximal horizontal coefficient:\t',num2str(param.max_coeff_horizontal)]);
j_disp(fname_log,['.. Maximal vertical coefficient:\t',num2str(param.max_coeff_vertical)]);
j_disp(fname_log,['.. Tolerance:\t\t\t\t',num2str(tolerance)]);

%% Convolution of the volume by sobel filters to obtain the gradient in each direction

% Creation of the sobel filters
factor_scales=scales/max(scales(1:3));

m_filters_sobel.x(1,:,:)=1/8*factor_scales(1)*[ 1  2  1;  2  4  2;  1  2  1];
m_filters_sobel.x(2,:,:)=1/8*factor_scales(1)*[ 0  0  0;  0  0  0;  0  0  0];
m_filters_sobel.x(3,:,:)=1/8*factor_scales(1)*[-1 -2 -1; -2 -4 -2; -1 -2 -1];

m_filters_sobel.y(:,1,:)=1/8*factor_scales(2)*[ 1  2  1;  2  4  2;  1  2  1];
m_filters_sobel.y(:,2,:)=1/8*factor_scales(2)*[ 0  0  0;  0  0  0;  0  0  0];
m_filters_sobel.y(:,3,:)=1/8*factor_scales(2)*[-1 -2 -1; -2 -4 -2; -1 -2 -1];

m_filters_sobel.z(:,:,1)=1/8*factor_scales(3)*[ 1  2  1;  2  4  2;  1  2  1];
m_filters_sobel.z(:,:,2)=1/8*factor_scales(3)*[ 0  0  0;  0  0  0;  0  0  0];
m_filters_sobel.z(:,:,3)=1/8*factor_scales(3)*[-1 -2 -1; -2 -4 -2; -1 -2 -1];

% Application of the sobel filters in x, y and z to compute the gradients of the volume
m_volume_sobel(:,:,:,1)=convn(m_volume,m_filters_sobel.x,'same');  %y gradient
m_volume_sobel(:,:,:,2)=convn(m_volume,m_filters_sobel.y,'same');  %x gradient
m_volume_sobel(:,:,:,3)=convn(m_volume,m_filters_sobel.z,'same');  %z gradient

% Normalization of the gradient
gradient_magnitude = sqrt(sum(m_volume_sobel.^2,4));
gradient_max = max(gradient_magnitude(:));
gradient_min = min(gradient_magnitude(:));

m_volume_sobel=m_volume_sobel/(gradient_max-gradient_min); %G=G/(Gmax-Gmin)

%% Iterative segmentation

j_disp(fname_log,['\n\n\nProcessing...']);

tic
% h=waitbar(0, 'Segmentation progression')
% Iteration loop

for i = 1:max([param.max_coeff_horizontal param.max_coeff_vertical])
    try
        % waitbar((i-1)/max([param.max_coeff_horizontal param.max_coeff_vertical]))
        % Incrementation of the horizontal and vertical coefficient
        if i > param.max_coeff_vertical
            coeff_vertical = param.max_coeff_vertical;
        else
            coeff_vertical=i;
        end
        
        if i > param.max_coeff_horizontal
            coeff_horizontal = param.max_coeff_horizontal;
        else
            coeff_horizontal=i;
        end
        j_disp(fname_log,['..Iteration: ', sprintf('%4.0f', i), ' || Horizontal coefficient: ', sprintf('%4.0f', coeff_horizontal), ' || Vertical coefficient:   ', sprintf('%4.0f', coeff_vertical)]);
        
        % Convergence loop
        while criteria > tolerance && nbIter < param.nbIterMax
            % Computation of the new radius and centerlines
            [m_radius m_norm_cord] = scs_radius_update(m_volume_sobel,m_center_line,m_tangent_cord,coord_radius,m_previous_radius,step,nom_radius,scales,angles,param);
            [m_radius_smoothed] = scs_smoothing(coeff_horizontal,coeff_vertical,m_radius);
            m_center_line_previous=m_center_line;
            [m_center_line, m_radius,m_tangent_cord] = scs_center_line_update(m_radius_smoothed,m_center_line_user,nom_radius,m_center_line_previous,angles,scales,param);
            [ratio, center_criteria, radius_criteria] = scs_convergence(m_center_line, m_center_line_previous, m_radius, m_previous_radius, param, m_volume, tolerance, nom_radius);
            
            % Display of the criterias
            j_disp(fname_log,['.... Loop: ' num2str(nbIter+1) '\t ||| ratio:   ',num2str(ratio,'%6.6f') '\t || Center criteria:   ',num2str(center_criteria,'%6.6f') '\t || Radius criteria:   ',num2str(radius_criteria,'%6.6f')]);
            
            % if only a few points don't converge, we update the coefficient
            % and/or order except if it is the last coefficient
            if ratio > param.ratio_criteria && coeff_horizontal ~= param.max_coeff_horizontal
                criteria = radius_criteria;
            elseif ratio <= param.ratio_criteria
                criteria = 0;
            end
            
            % Save the previous_radius and increment to the next iteration
            m_previous_radius = m_radius;
            nbIter = nbIter+1;
        end
        
        % Reset parameter
        criteria = 10;
        nbIter=0;
        
        % Save the radius and centerline for all the slices throughout iterations
        s_save.radius(i,:,:)=m_radius;
        s_save.centerline(i,:,:)=m_center_line(:,:);
        
        % Computing the the cross_sectional area for each slice for this
        % iteration
        [average_area(i),area_per_slice(i,:),volume_tot(i),length_tot(i)] = scs_measurements(squeeze(s_save.centerline(i,:,:)), scales, squeeze(s_save.radius(i,:,:)), param.slices(1), param.slices(2), angles);
        
    catch exception
        s_save.angles=angles;
        s_save.average_area=average_area;
        s_save.area_per_slice=area_per_slice;
        s_save.volume_tot=volume_tot;
        s_save.lenght_tot=length_tot;
        s_save.m_nifti=m_nifti;
        s_save.scales=scales;
        s_save.resampling=param.resampling;
        s_save.slices=param.slices;
        s_save.tangent_cord=m_tangent_cord;
        s_save.norm_cord=m_norm_cord;
        s_save.param=param;
        s_save.fname_log=fname_log;
        save([output(1:end-4) '_Failed' output(end-3:end)],'-struct','s_save');
        disp(['Spinal cord segmentation failed for ' input '. Reason: ' exception.message])
        break;
    end
end

j_disp(fname_log,['\n\nSegmentation finished in ',num2str(toc), ' seconds.']);

%% Saving the results in the output file

s_save.angles=angles;
s_save.average_area=average_area;
s_save.area_per_slice=area_per_slice;
s_save.volume_tot=volume_tot;
s_save.lenght_tot=length_tot;
s_save.m_nifti=m_nifti;
s_save.scales=scales;
s_save.resampling=param.resampling;
s_save.slices=param.slices;
s_save.tangent_cord=m_tangent_cord;
s_save.norm_cord=m_norm_cord;
s_save.param=param;
s_save.fname_log=fname_log;

try
    save(output,'-struct','s_save');
    j_disp(fname_log,['\n\nOutput file contaning all the matrices saved.']);
catch exception
    new_output = [pwd '/failed_save'];
    save(new_output,'-struct','s_save');
    disp(['Impossible to save in the desired output file. Save as been done here: ' new_output])
end

%  profile off % to stop the profiler analysis
%  p = profile('info');
%  save myprofiledata strcat(scs_profile_T datestr(now, '_yyyy_mm_dd'))
%  clear p
%  load myprofiledata
%  profview(0,p)

end

