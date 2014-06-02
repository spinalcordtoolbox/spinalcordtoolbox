function scs_display(m_nifti, m_center_line, m_radius, start_z, end_z, angles, resampling, area_per_slice, average_area)
% scs_display 
%
% SYNTAX:
% [] = scs_display(M_NIFTI, M_CENTER_LINE, M_RADIUS, START_Z, END_Z, ANGLES, RESAMPLING, AREA_PER_SLICE, AVERAGE_AREA)
%
% _________________________________________________________________________
% INPUTS:
%
% M_NIFTI
% 	(XxYxZ array) Voxels intensity of raw NIfTI image
%
% M_CENTER_LINE
% 	(Nx3 array) Coordinates of the spinal cord for each slice of M_VOLUME
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
% ANGLES
%   Angles used to compute the radius
%
% RESAMPLING
%	Resampling factor fo the image
%
% AREA_PER_SLICE
%	Area of the cross-sectional surface for each slice (mm2)
%
% AVERAGE_AREA
%	Mean area of all the cross-sectional surfaces (mm2)
% _________________________________________________________________________
% OUTPUTS:
%
% NONE

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%   Initialization
nb_iterations = size(m_radius,1);
nb_slices = size(m_radius,2);
nb_angles = length(angles);

% display of the contour
% retrieve the original m_volume (remember that the slices have been padded
%   in the m_volume before)
m_volume=m_nifti;
m_volume=imresize(m_volume,resampling);
m_volume(:,:,end_z+1:size(m_volume,3))=[];  % Suppression of the last slices
m_volume(:,:,1:start_z)=[];                 % Suppression of the first slices

% Compute the contours for all the iterations
% contour(nb_angles, x and y, nb_slices, nb_iterations)
contour = zeros(nb_angles,2,nb_slices, nb_iterations);
for j=1:nb_iterations
    for i = 1 : nb_slices
        [x,y] = pol2cart(angles,squeeze(m_radius(j,i,:))');
        contour(:,:,i,j) = [x' y'];               % save contour for visualization except for the two last slices
    end
end

scrsz = get(0,'ScreenSize');         % full screen

% Display of the final center_line
m_center_line_user=squeeze(m_center_line(1,:,:))';
m_center_line_final = squeeze(m_center_line(end,:,:))';

f=figure('Name','Position of the computed centerline (red) and the user defined centerline (green)');
set(f,'Position',[1 1 scrsz(3) scrsz(4)]);
set(f,'color','w')

% xz slice at the middle of the spine
subplot(2,3,[4,5])
img_buffer=squeeze(m_volume(:,round(mean(m_center_line_final(:,2))),:));
imagesc(img_buffer'); colormap gray, axis image
hold on
plot(m_center_line_final(:,1),m_center_line_final(:,3),'r',m_center_line_user(:,1),m_center_line_user(:,3),'g--')
ylim=get(gca,'YLim');xlim=get(gca,'XLim'); ftsize = mean(get(f,'position'))/50;
text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',ftsize)
text((xlim(1)+xlim(2))/2,ylim(1), 'S', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
text((xlim(1)+xlim(2))/2,ylim(2), 'I', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
title('sagittal view','fontsize',16)
xlabel('x','fontsize',12)
ylabel('z','fontsize',12)
hold off
% yz slice 
subplot(2,3,[3,6])
img_buffer=squeeze(m_volume(round(mean(m_center_line_final(:,1))),:,:));
imagesc(img_buffer'); colormap gray, axis image
hold on
plot(m_center_line_final(:,2),m_center_line_final(:,3),'r',m_center_line_user(:,2),m_center_line_user(:,3),'g--')
ylim=get(gca,'YLim');xlim=get(gca,'XLim'); ftsize = mean(get(f,'position'))/50;
text(xlim(1),(ylim(1)+ylim(2))/2, 'S', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
text(xlim(2),(ylim(1)+ylim(2))/2, 'I', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',ftsize)
text((xlim(1)+xlim(2))/2,ylim(1), 'R', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
text((xlim(1)+xlim(2))/2,ylim(2), 'L', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)

title('coronal view','fontsize',16)
xlabel('y','fontsize',12)
ylabel('z','fontsize',12)
hold off
% xy slice 
subplot(2,3,[1,2])
img_buffer=squeeze(m_volume(:,:,round(mean(m_center_line_final(:,3)))));
imagesc(img_buffer'); colormap gray, axis image
hold on
plot(m_center_line_final(:,1),m_center_line_final(:,2),'r',m_center_line_user(:,1),m_center_line_user(:,2),'g--')
ylim=get(gca,'YLim');xlim=get(gca,'XLim'); ftsize = mean(get(f,'position'))/50;
text(xlim(1),(ylim(1)+ylim(2))/2, 'R', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
text(xlim(2),(ylim(1)+ylim(2))/2, 'L', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',ftsize)
text((xlim(1)+xlim(2))/2,ylim(1), 'P', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
text((xlim(1)+xlim(2))/2,ylim(2), 'A', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
title('axial view','fontsize',16)
xlabel('x','fontsize',12)
ylabel('y','fontsize',12)
hold off

scs_slider(m_volume, m_center_line, contour, area_per_slice, average_area)


end


