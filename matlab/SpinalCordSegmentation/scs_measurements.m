function [average_area,area_sum,volume_tot,length_tot] = scs_measurements(m_center_line, scales, m_radius, start_z, end_z, theta)
% scs_measurements
%   Description
%
% SYNTAX:
% [AVERAGE_AREA,AREA_SUM,VOLUME_TOT,LENGTH_TOT] = scs_measurements(M_NIFTI, M_CENTER_LINE, SCALES, M_RADIUS, START_Z, END_Z, THETA, RESAMPLING, M_CENTER_LINE_USER)
%
% _________________________________________________________________________
% INPUTS:
%
% M_NIFTI    
%   (XxYxZ array) Voxels intensity of raw NIfTI image
%
% M_CENTER_LINE
%   (Nx3 array) Coordinates of the spinal cord for each slice of M_VOLUME
%
% SCALES
%   (4x1 array) Scales in mm of X, Y and Z axes in M_VOLUME
%
% M_RADIUS
%   (2D matrix) Value of the radius for each angles and each slice
%   of the splinal cord 
%
% START_Z
%	First slice of the field of view
%
% END_Z
%	Last slice of the field of view
%
% THETA
%   Angles used to compute the radius
%
% RESAMPLING
%	Resampling factor fo the image
%
% _________________________________________________________________________
% OUTPUTS:
%
% AVERAGE_AREA
%   Mean area of all the cross-sectional surfaces (mm2)
%
% AREA_SUM
%	Area of the cross-sectional surface for each slice (mm2)
%
% VOLUME_TOT
%   Total volume of the field of view (mm3)
%
% LENGTH_TOT
%	Length of the spinal cord in the field of view (mm)   
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global fname_log

% size_vector equals the number of slices
size_vector = size(m_center_line,2);

%   Computation of the derivatives in mm
dCx=(m_center_line(1,2:end)-m_center_line(1,1:end-1))*scales(1);
dCy=(m_center_line(2,2:end)-m_center_line(2,1:end-1))*scales(2);
dCz=ones(size(dCx,2),1)*scales(3);

dC = [dCx' dCy' dCz];

%   Computation of the norm
norm_tangent=sqrt(sum(dC.*dC, 2));
  
%   Compute the local length 
for i = 1:size(norm_tangent,1)
    length_local(i) = trapz(norm_tangent(1:i));
end
length_tot = length_local(end);

%   Compute the total area
P1=[0 0];
contour = zeros(size(m_radius,2),2,size(m_radius,1));
for i = 1 : size_vector              
    [x,y] = pol2cart(theta,m_radius(i,:));
    contour(:,:,i) = [x' y'];
    x = x*scales(1);
    y = y*scales(2);
   % computation of the coordinates
    for j = 1:size(m_radius,2)
        if j == size(m_radius,2)
            P2=[x(j) y(j)];
            P3=[x(1) y(1)];
        else
            P2=[x(j) y(j)];
            P3=[x(j+1) y(j+1)];
        end
        area(j) = triangle_area([P1;P2;P3]);
    end
    area_sum(i) = sum(area);
end             

volume_tot = trapz(length_local,area_sum(1:size(length_local,2)));

average_area = volume_tot/length_tot;

%Display of parameters
j_disp(fname_log,'..Measurements for this iteration:');
j_disp(fname_log,['... Volume of VOI (mm3):          ',num2str(volume_tot)]);
j_disp(fname_log,['... Length of VOI (mm) :          ',num2str(length_tot)]);
j_disp(fname_log,['... Average area  (mm2):          ',num2str(average_area)]);
j_disp(fname_log,['\n\n']);

end


