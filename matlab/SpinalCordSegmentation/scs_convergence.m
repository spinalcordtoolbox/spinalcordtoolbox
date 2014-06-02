function [ratio, center_criteria, radius_criteria] = scs_convergence(m_center_line, m_center_line_previous, m_radius, m_previous_radius, param, m_volume, tolerance, nom_radius)
% scs_convergence 
%   This function computes the convergence criteria for the
%   spinal_cord_segmentation algorithm.
%
% SYNTAX:
%   [RATIO, CENTER_CRITERIA, RADIUS_CRITERIA] = scs_convergence(M_CENTER_LINE, M_CENTER_LINE_PREVIOUS, M_RADIUS, M_PREVIOUS_RADIUS, PARAM, M_VOLUME, TOLERANCE, NOM_RADIUS)
%
% _________________________________________________________________________
% INPUTS:  
%
% M_CENTER_LINE
%   (Nx3 array) Coordinates of the spinal cord for each slice (N) of
%
% M_CENTER_LINE_PREVIOUS
%   m_center_line of the previous iteration   
%
% M_RADIUS
%   (2D matrix) Value of the radius for each angles and each slice
%   of the splinal cord 
%
% M_PREVIOUS_RADIUS
%   m_radius of the previous iteration
% 
% PARAM
%   Struct containing an assortment of user defined parameters
%
% M_VOLUME
%   (XxYxZ array) Voxels intensity of desired NIfTI image
%
% TOLERANCE
%   Maximal value that a radius can move to be considered as converged
% 
% NOM_RADIUS
%   Nominal radius
% _________________________________________________________________________
% OUTPUTS:
%
% RATIO
%   percentage of radius that meet the tolerance factor as define by
%   the user
% 
% CENTER_CRITERIA
%   Maximal displacement of a center_line for this iteration
%
% RADIUS_CRITERIA
%   Maximal displacement of a radius for this iteration
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Computation of the convergence criterias
center_criteria = max(max(abs(m_center_line-m_center_line_previous)));
radius_criteria = max(max(abs(m_radius-m_previous_radius)));

% Verification that the centerline is within the volume
if min(min(m_center_line)) < 1 % no negative coordinates
    scs_warning_error(219, param)
elseif max(m_center_line(1,:)) > size(m_volume,1) || max(m_center_line(2,:)) > size(m_volume,2) || max(m_center_line(3,:)) > size(m_volume,3)   % no coordinates out of the volume
    scs_warning_error(219, param)
end

% Verification that the radius values are plausible
if radius_criteria > param.nom_radius || center_criteria > param.nom_radius
    scs_warning_error(214, param)
elseif max(max(m_radius)) > 3*nom_radius
    scs_warning_error(109, param)
elseif min(min(m_radius)) < nom_radius/3
    scs_warning_error(110, param)
elseif max(max(m_radius)) > 5*nom_radius
    scs_warning_error(215, param)
elseif min(min(m_radius)) <=0
    scs_warning_error(216, param)
end

% Computation of the ratio of radius that converged (set
% param.ratio_criteria = 0 to use the radius_criteria instead as the
% convergence criteria instead)
absolute_error = abs(m_radius-m_previous_radius);
nbr_resisted_point = length(find(absolute_error>tolerance));
nbr_point = length(absolute_error(:));
ratio = nbr_resisted_point/nbr_point;

end

