function [] = scs_nifti_save_v4(output1,output2,output3,output_straightnened,param)
% scs_nifti_save
% This function generates binary nifti images. 
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
%   segmentation
% _________________________________________________________________________
% OUTPUTS:
%
% NONE

% ------------------------------------------------------------------------
% Initialization
% ------------------------------------------------------------------------

load(output1)

%center line of the last iteration
cl_last_iter = squeeze(centerline(end, :, :)); 
%change of origin
cl_last_iter(3,:) = cl_last_iter(3,:)+param.slices(1)-1;

radius_last_iter = squeeze(radius(end, :, :));
contour = zeros(size(angles,2),2,size(centerline,3));
for i = 1 : size(centerline,3)              
    [x,y] = pol2cart(angles,radius_last_iter(i,:));
    x = x + cl_last_iter(1,i);
    y = y + cl_last_iter(2,i);
    contour(:,:,i) = [x' y'];     
end

%initialization of the nifti output of the center line
m_cl = zeros(size(m_nifti,1),size(m_nifti,2),size(m_nifti,3)); 

m_surface = m_cl;

% ------------------------------------------------------------------------
% Saving the centerline
% ------------------------------------------------------------------------

% Create a binary matrix of the same size of the raw image for the centerline
% (note that the centerline is in axial view)
for i=1:size(cl_last_iter,2)
    m_cl(round(cl_last_iter(1,i)),round(cl_last_iter(2,i)),round(cl_last_iter(3,i)))=1;
end

% Save of the centerline in a NIfTI image
CL=param.nii;
CL.img=logical(m_cl); CL.hdr.dime.bitpix=1;
save_nii_v2(CL, output2);

j_disp(fname_log,['\n\nCenterline saved in NIfTI.']);


% ------------------------------------------------------------------------
% Saving the segmented cord surface
% ------------------------------------------------------------------------

% Create a binary matrix of the same size of the raw image for the surface
m_surface = false(size(m_surface));

for i=1:size(centerline,3)
    m_surface(:,:,round(cl_last_iter(3,i))) = poly2mask(contour(:,2,i),contour(:,1,i),size(m_surface,1),size(m_surface,2));
end

SF=param.nii;
SF.img=logical(m_surface); SF.hdr.dime.bitpix=1;
% Save of the surface in a NIfTI image
save_nii_v2(SF, output3);

j_disp(fname_log,['\nSegmented surfaces saved in NIfTI.']);



% ------------------------------------------------------------------------
% Saving the straightened data
% ------------------------------------------------------------------------
load(output_straightnened);
ST=param.nii;
ST.img=m_straight;
save_nii_v2(ST,output_straightnened)

% ------------------------------------------------------------------------
% Saving the reoriented data
% ------------------------------------------------------------------------
save_nii_v2(param.nii,[param.nii.fileprefix '_reorient' param.ext])

end

