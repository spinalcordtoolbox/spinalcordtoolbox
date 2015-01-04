function [] = scs_nifti_save_v3(input1,output1,output2,output3,output_straightnened,param)
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

fsloutput = ['export FSLOUTPUTTYPE=NIFTI;'];

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


m_nifti=permute(m_nifti, [1 3 2]);
scales([1 2 3]) = scales([1 3 2]);

%initialization of the nifti output of the center line
m_cl = zeros(size(m_nifti,1),size(m_nifti,2),size(m_nifti,3)); 

m_surface = m_cl;

x = param.orient_init(1);
y = param.orient_init(2);
z = param.orient_init(3);

if strcmp(x,'A'), x='AP'; end
if strcmp(x,'P'), x='PA'; end
if strcmp(x,'S'), x='SI'; end
if strcmp(x,'I'), x='IS'; end
if strcmp(x,'R'), x='RL'; end
if strcmp(x,'L'), x='LR'; end

if strcmp(y,'A'), y='AP'; end
if strcmp(y,'P'), y='PA'; end
if strcmp(y,'S'), y='SI'; end
if strcmp(y,'I'), y='IS'; end
if strcmp(y,'R'), y='RL'; end
if strcmp(y,'L'), y='LR'; end

if strcmp(z,'A'), z='AP'; end
if strcmp(z,'P'), z='PA'; end
if strcmp(z,'S'), z='SI'; end
if strcmp(z,'I'), z='IS'; end
if strcmp(z,'R'), z='RL'; end
if strcmp(z,'L'), z='LR'; end

% ------------------------------------------------------------------------
% Saving the centerline
% ------------------------------------------------------------------------

% Create a binary matrix of the same size of the raw image for the centerline
% (note that the centerline is in axial view)
for i=1:size(centerline,3)
    m_cl(round(cl_last_iter(1,i)),round(cl_last_iter(3,i)),round(cl_last_iter(2,i)))=1;
end

% Save of the centerline in a NIfTI image
save_avw_v2(m_cl, output2, 'f', scales(1:3));

input_reorient=[input1,'_reorient'];

%copy the header of reorient data in the centerline nifti
cmd = [fsloutput,' fslcpgeom ', input_reorient, ' ', output2, ' -d'];
[status result] = unix(cmd); if status, error(result); end

% change the orientation to match the initial orientation
if param.det_qform>0
    cmd=[fsloutput,'fslswapdim ',output2,' -x y z ',output2];
    [status result] = unix(cmd); if status, error(result); end
    
    cmd=[fsloutput,' fslorient -swaporient ',output2];
    [status result]=unix(cmd); if status, error(result); end
    
    cmd=[fsloutput,' fslswapdim ',output2, ' ',x, ' ', y, ' ', z, ' ',output2];
    [status result]=unix(cmd); if status, error(result); end
    
else
    
    cmd=[fsloutput,' fslswapdim ',output2, ' ',x, ' ', y, ' ', z, ' ',output2];
    [status result]=unix(cmd); if status, error(result); end
    
end

%copy the initial header to the centerline data
cmd = [fsloutput,' fslcpgeom ', input1, ' ', output2, ' -d'];
[status result] = unix(cmd); if status, error(result); end



j_disp(fname_log,['\n\nCenterline saved in NIfTI.']);


% ------------------------------------------------------------------------
% Saving the segmented cord surface
% ------------------------------------------------------------------------

% Create a binary matrix of the same size of the raw image for the surface
for i=1:size(centerline,3)
    for j = 1:size(angles,2)
        m_surface(ceil(contour(j,1,i)),round(cl_last_iter(3,i)),ceil(contour(j,2,i)))=1;
    end
end

% Create a binary matrix of the same size of the raw image for the surface
for i=1:size(centerline,3)
    for j = 1:size(angles,2)
        m_surface(floor(contour(j,1,i)),round(cl_last_iter(3,i)),floor(contour(j,2,i)))=1;
    end
end


% Filling the contours
for i=1:size(m_surface,1), m_surface(i,:,:) = imfill(squeeze(m_surface(i,:,:))); end
for k=1:size(m_surface,3), m_surface(:,:,k) = imfill(squeeze(m_surface(:,:,k))); end
for j=1:size(m_surface,2), m_surface(:,j,:) = imfill(squeeze(m_surface(:,j,:))); end

se = strel('disk',3);

% Perform an opening
m_surface2 = imopen(m_surface,se);
m_surface3 = imreconstruct(m_surface2,m_surface);

% Filling the contours
for i=1:size(m_surface,1), m_surface3(i,:,:) = imfill(squeeze(m_surface3(i,:,:))); end
for k=1:size(m_surface,3), m_surface3(:,:,k) = imfill(squeeze(m_surface3(:,:,k))); end
for j=1:size(m_surface,2), m_surface3(:,j,:) = imfill(squeeze(m_surface3(:,j,:))); end
% Filling the contours
for i=1:size(m_surface,1), m_surface3(i,:,:) = imfill(squeeze(m_surface3(i,:,:))); end
for k=1:size(m_surface,3), m_surface3(:,:,k) = imfill(squeeze(m_surface3(:,:,k))); end
for j=1:size(m_surface,2), m_surface3(:,j,:) = imfill(squeeze(m_surface3(:,j,:))); end

m_surface3 = imclose(m_surface3,se);


clear m_surface m_surface2

% Save of the surface in a NIfTI image
save_avw_v2(m_surface3, output3, 'f', scales(1:3));

%copy the header of reorient data in the surface image
cmd = [fsloutput,'fslcpgeom ', input_reorient, ' ', output3, ' -d'];
[status result] = unix(cmd); if status, error(result); end

% change the orientation to match the initial orientation
if param.det_qform>0
    cmd=[fsloutput,'fslswapdim ',output3,' -x y z ',output3];
    [status result] = unix(cmd); if status, error(result); end
    
    cmd=[fsloutput,' fslorient -swaporient ',output3];
    [status result]=unix(cmd); if status, error(result); end
    
    cmd=[fsloutput,' fslswapdim ',output3, ' ',x, ' ', y, ' ', z, ' ',output3];
    [status result]=unix(cmd); if status, error(result); end
    
else
    
    cmd=[fsloutput,' fslswapdim ',output3, ' ',x, ' ', y, ' ', z, ' ',output3];
    [status result]=unix(cmd); if status, error(result); end
    
end

%copy the initial header to the surface nifti
cmd = [fsloutput,'fslcpgeom ', input1, ' ', output3, ' -d'];
[status result] = unix(cmd); if status, error(result); end

j_disp(fname_log,['\nSegmented surfaces saved in NIfTI.']);



% ------------------------------------------------------------------------
% Saving the straightened data
% ------------------------------------------------------------------------
load(output_straightnened);
save_avw_v2(m_straight,output_straightnened,'f',scales(1:3),input_reorient);

% change the orientation to match the initial orientation
if param.det_qform>0
    cmd=[fsloutput,'fslswapdim ',output_straightnened,' -x y z ',output_straightnened];
    [status result] = unix(cmd); if status, error(result); end
    
    cmd=[fsloutput,' fslorient -swaporient ',output_straightnened];
    [status result]=unix(cmd); if status, error(result); end
    
    cmd=[fsloutput,' fslswapdim ',output_straightnened, ' ',x, ' ', y, ' ', z, ' ',output_straightnened];
    [status result]=unix(cmd); if status, error(result); end
    
else
    
    cmd=[fsloutput,' fslswapdim ',output_straightnened, ' ',x, ' ', y, ' ', z, ' ',output_straightnened];
    [status result]=unix(cmd); if status, error(result); end
    
end

%copy the initial header to the centerline data
cmd = [fsloutput,' fslcpgeom ', input1, ' ', output_straightnened, ' -d'];
[status result] = unix(cmd); if status, error(result); end

% delete file
delete([input_reorient,'.nii']);

end

