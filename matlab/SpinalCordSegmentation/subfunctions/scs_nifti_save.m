function [] = scs_nifti_save(input1,output1,output2,output3,param)
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

cl_last_iter = squeeze(centerline(end, :, :)); %center line of the last iteration
cl_last_iter(3,:) = cl_last_iter(3,:)+param.slices(1)-1;

radius_last_iter = squeeze(radius(end, :, :));
contour = zeros(size(angles,2),2,size(centerline,3));
for i = 1 : size(centerline,3)              
    [x,y] = pol2cart(angles,radius_last_iter(i,:));
    x = x + cl_last_iter(1,i);
    y = y + cl_last_iter(2,i);
    contour(:,:,i) = [x' y'];     
end

% m_cl = zeros(size(m_nifti,1),size(m_nifti,2),size(m_nifti,3)); %initialization of the nifti output of the center line

% % Repermurte to run the code
% m_nifti=permute(m_nifti, [2 1 3]);
% scales([1 2 3]) = scales([2 1 3]);
% 
% % Permute to fit with the initial image
% param.orient_new([1 2 3]) = permute(param.orient_new,param.permute);
% 
% % if param.orient_new(1) == 'R' && param.orient_new(1) ~= param.orient_init(1), m_nifti = flipdim(m_nifti,2); cl_last_iter(1,:) = size(m_cl,1)+1 - cl_last_iter(1,:); contour(:,1,:) = size(m_cl,1)+1 - contour(:,1,:); test(1)='L'; end
% % if param.orient_new(1) == 'P' && param.orient_new(1) ~= param.orient_init(1), m_nifti = flipdim(m_nifti,1); cl_last_iter(2,:) = size(m_cl,2)+1 - cl_last_iter(2,:); contour(:,2,:) = size(m_cl,2)+1 - contour(:,2,:); test(1)='A'; end
% % if param.orient_new(1) == 'S' && param.orient_new(1) ~= param.orient_init(1), m_nifti = flipdim(m_nifti,3); cl_last_iter(3,:) = size(m_cl,3)+1 - cl_last_iter(3,:); contour = flipdim(contour,3); test(1)='I'; end
% % 
% % if param.orient_new(2) == 'P' && param.orient_new(2) ~= param.orient_init(2), m_nifti = flipdim(m_nifti,1); cl_last_iter(2,:) = size(m_cl,2)+1 - cl_last_iter(2,:); contour(:,2,:) = size(m_cl,2)+1 - contour(:,2,:); test(2)='A'; end
% % if param.orient_new(2) == 'R' && param.orient_new(2) ~= param.orient_init(2), m_nifti = flipdim(m_nifti,2); cl_last_iter(1,:) = size(m_cl,1)+1 - cl_last_iter(1,:); contour(:,1,:) = size(m_cl,1)+1 - contour(:,1,:); test(2)='L'; end
% % if param.orient_new(2) == 'S' && param.orient_new(2) ~= param.orient_init(2), m_nifti = flipdim(m_nifti,3); cl_last_iter(3,:) = size(m_cl,3)+1 - cl_last_iter(3,:); contour = flipdim(contour,3); test(2)='I'; end
% % 
% % if param.orient_new(3) == 'S' && param.orient_new(3) ~= param.orient_init(3), m_nifti = flipdim(m_nifti,3); cl_last_iter(3,:) = size(m_cl,3)+1 - cl_last_iter(3,:); contour = flipdim(contour,3); test(3)='I'; end
% % if param.orient_new(3) == 'P' && param.orient_new(3) ~= param.orient_init(3), m_nifti = flipdim(m_nifti,1); cl_last_iter(2,:) = size(m_cl,2)+1 - cl_last_iter(2,:); contour(:,2,:) = size(m_cl,2)+1 - contour(:,2,:); test(3)='A'; end
% % if param.orient_new(3) == 'R' && param.orient_new(3) ~= param.orient_init(3), m_nifti = flipdim(m_nifti,2); cl_last_iter(1,:) = size(m_cl,1)+1 - cl_last_iter(1,:); contour(:,1,:) = size(m_cl,1)+1 - contour(:,1,:); test(3)='L'; end
% 
% if param.orient_new(1) == 'R' && param.orient_new(1) ~= param.orient_init(1),
%     m_nifti = flipdim(m_nifti,2); 
%     cl_last_iter(1,:) = size(m_cl,1)+1 - cl_last_iter(1,:); 
%     contour(:,2,:) = size(m_cl,2)+1 - contour(:,2,:); 
%     test(1)='L'; 
% end
% 
% if param.orient_new(1) == 'P' && param.orient_new(1) ~= param.orient_init(1), 
%     m_nifti = flipdim(m_nifti,1); 
%     cl_last_iter(2,:) = size(m_cl,2)+1 - cl_last_iter(2,:);
%     contour(:,1,:) = size(m_cl,1)+1 - contour(:,1,:); 
%     test(1)='A'; 
% end
% if param.orient_new(1) == 'S' && param.orient_new(1) ~= param.orient_init(1), 
%     m_nifti = flipdim(m_nifti,3); 
%     cl_last_iter(3,:) = size(m_cl,3)+1 - cl_last_iter(3,:); 
%     contour = flipdim(contour,3); 
%     test(1)='I'; 
% end
% 
% if param.orient_new(2) == 'P' && param.orient_new(2) ~= param.orient_init(2), 
%     m_nifti = flipdim(m_nifti,1); 
%     cl_last_iter(2,:) = size(m_cl,2)+1 - cl_last_iter(2,:); 
%     contour(:,1,:) = size(m_cl,1)+1 - contour(:,1,:); 
%     test(2)='A'; 
% end
% if param.orient_new(2) == 'R' && param.orient_new(2) ~= param.orient_init(2), 
%     m_nifti = flipdim(m_nifti,2); 
%     cl_last_iter(1,:) = size(m_cl,1)+1 - cl_last_iter(1,:); 
%     contour(:,2,:) = size(m_cl,2)+1 - contour(:,2,:); 
%     test(2)='L'; 
% end
% if param.orient_new(2) == 'S' && param.orient_new(2) ~= param.orient_init(2),
%    m_nifti = flipdim(m_nifti,3); 
%    cl_last_iter(3,:) = size(m_cl,3)+1 - cl_last_iter(3,:); 
%    contour = flipdim(contour,3); 
%    test(2)='I'; 
% end
% 
% if param.orient_new(3) == 'S' && param.orient_new(3) ~= param.orient_init(3), 
%     m_nifti = flipdim(m_nifti,3); 
%     cl_last_iter(3,:) = size(m_cl,3)+1 - cl_last_iter(3,:); 
%     contour = flipdim(contour,3); 
%     test(3)='I'; 
% end
% if param.orient_new(3) == 'P' && param.orient_new(3) ~= param.orient_init(3),
%    m_nifti = flipdim(m_nifti,1); 
%    cl_last_iter(2,:) = size(m_cl,2)+1 - cl_last_iter(2,:); 
%    contour(:,1,:) = size(m_cl,1)+1 - contour(:,1,:); 
%    test(3)='A'; 
% end
% if param.orient_new(3) == 'R' && param.orient_new(3) ~= param.orient_init(3), 
%     m_nifti = flipdim(m_nifti,2); 
%     cl_last_iter(1,:) = size(m_cl,1)+1 - cl_last_iter(1,:); 
%     contour(:,2,:) = size(m_cl,2)+1 - contour(:,2,:); 
%     test(3)='L'; end
% 
% 
% m_nifti=permute(m_nifti, param.permute);
% m_nifti=permute(m_nifti, param.permute);
% scales([1 2 3]) = scales(param.permute);
% scales([1 2 3]) = scales(param.permute);
m_nifti=permute(m_nifti, [1 3 2]);
scales([1 2 3]) = scales([1 3 2]);

m_cl = zeros(size(m_nifti,1),size(m_nifti,2),size(m_nifti,3)); %initialization of the nifti output of the center line

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
save_avw(m_cl, output2, 'f', scales(1:3));

input_reorient=[input1,'_reorient'];

cmd = ['export FSLOUTPUTTYPE=NIFTI; fslcpgeom ', input_reorient, ' ', output2, ' -d'];
[status result] = unix(cmd); if status, error(result); end

cmd=['fslswapdim ',output2, ' ',x, ' ', y, ' ', z, ' ',output2];
[status result]=unix(cmd); if status, error(result); end

cmd = ['fslcpgeom ', input1, ' ', output2, ' -d'];
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

% for i=1:size(m_surface,1), m_surface3(i,:,:) = imopen(squeeze(m_surface3(i,:,:)),se); end
% for k=1:size(m_surface,3), m_surface3(:,:,k) = imopen(squeeze(m_surface3(:,:,k)),se); end
% for j=1:size(m_surface,2), m_surface3(:,j,:) = imopen(squeeze(m_surface3(:,j,:)),se); end
% % Create a binary matrix of the same size of the raw image for the surface
% for i=1:size(centerline,3)
%     for j = 1:size(angles,2)
%         m_surface(floor(contour(j,1,i)),round(cl_last_iter(3,i)),floor(contour(j,2,i)))=1;
%     end
% end

% % Filling the contours
% for i=size(m_surface,1):1, m_surface(i,:,:) = imfill(squeeze(m_surface(i,:,:))); end
% for k=size(m_surface,3):1, m_surface(:,:,k) = imfill(squeeze(m_surface(:,:,k))); end
% for j=size(m_surface,2):1, m_surface(:,j,:) = imfill(squeeze(m_surface(:,j,:))); end

% % Create a binary matrix of the same size of the raw image for the surface
% for i=1:size(centerline,3)
%     for j = 1:size(angles,2)
%         m_surface(floor(contour(j,1,i)),round(cl_last_iter(3,i)),floor(contour(j,2,i)))=1;
%     end
% end
% 
% % Filling the contours
% for i=1:size(m_surface,1), m_surface(i,:,:) = imfill(squeeze(m_surface(i,:,:)),'holes'); end
% for k=1:size(m_surface,3), m_surface(:,:,k) = imfill(squeeze(m_surface(:,:,k)),'holes'); end
% for j=1:size(m_surface,2), m_surface(:,j,:) = imfill(squeeze(m_surface(:,j,:)),'holes'); end

clear m_surface m_surface2

% Save of the surface in a NIfTI image
save_avw(m_surface3, output3, 'f', scales(1:3));

cmd = ['fslcpgeom ', input_reorient, ' ', output3, ' -d'];
[status result] = unix(cmd); if status, error(result); end

cmd=['fslswapdim ',output3, ' ',x, ' ', y, ' ', z, ' ',output3];
[status result]=unix(cmd); if status, error(result); end

cmd = ['fslcpgeom ', input1, ' ', output3, ' -d'];
[status result] = unix(cmd); if status, error(result); end

j_disp(fname_log,['\nSegmented surfaces saved in NIfTI.']);

end

