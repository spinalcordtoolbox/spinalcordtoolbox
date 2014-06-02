%% video

clear all
clc
close all
% input = '/Users/gbm4900/scs_results/surface_43.nii'
% [status result] = unix(['fslhd ' input]);
% %Plane view
% % Read the orientation of the image
% orientation{1} = strtrim(result(findstr(result,'qform_xorient')+13:findstr(result,'qform_yorient')-1));
% orientation{2} = strtrim(result(findstr(result,'qform_yorient')+13:findstr(result,'qform_zorient')-1));
% orientation{3} = strtrim(result(findstr(result,'qform_zorient')+13:findstr(result,'sform_name')-1));
%     
% % Only keeps the first letter of the orient structure
% %   where each letter corresponds to :
% % R: Right-to-Left         L: Left-to-Right
% % P: Posterior-to-Anterior A: Anterior-to-Posterior
% % I: Inferior-to-Superior  S: Superior-to-Inferior
% orientation{1} = orientation{1}(1);
% orientation{2} = orientation{2}(1);
% orientation{3} = orientation{3}(1);
%     
% % Save initial orientation
% param.orient_init = cell2mat(orientation);
% [m_nifti,dims,scales,bpp,endian] = scs_read_avw(input);



input = '/Users/gbm4900/code/tests/T15/scs_results/43_space_2013_03_26_Test03';
load(input);
%   Initialization
nb_iterations = size(radius,1);
nb_slices = size(radius,2);
nb_angles = length(angles);

% display of the contour
% retrieve the original m_volume (remember that the slices have been padded
%   in the m_volume before)
m_volume=m_nifti;
m_volume=imresize(m_volume,resampling);
m_volume(:,:,param.slices(2)+1:size(m_volume,3))=[];  % Suppression of the last slices
m_volume(:,:,1:param.slices(1))=[];                 % Suppression of the first slices

% Compute the contours for all the iterations
% contour(nb_angles, x and y, nb_slices, nb_iterations)
contour = zeros(nb_angles,2,nb_slices, nb_iterations);
for j=1:nb_iterations
    for i = 1 : nb_slices
        [x,y] = pol2cart(angles,squeeze(radius(j,i,:))');
        contour(:,:,i,j) = [x' y'];               % save contour for visualization except for the two last slices
    end
end
j =1;
%%
%scrsz = get(0,'ScreenSize');   
f = figure
%set(f,'Position',[1 1 scrsz(3) scrsz(4)]);
while (j ==1)
writerObj = VideoWriter('SegmentationFinal2.avi','Uncompressed AVI')
writerObj.FrameRate = 10;
open(writerObj)
scrsz = get(0,'ScreenSize');
%f=figure
set(f,'Position',[1 1 scrsz(3) scrsz(4)]/1.8);
set(f,'color','w')
figure_size = get(f,'position');
ftsize = 16;%mean(figure_size(3:4))/100; 	
ha = tight_subplot(1,2,[.005 .005],[.3 .3],[.05 .05])
%%
for i = 1: size(centerline,3)
    axes(ha(1))
    % find the pixel coordinate of the contour
    img_contour_x = contour(:,1,i,end)+centerline(end,1,i);
    img_contour_y = contour(:,2,i,end)+centerline(end,2,i);
    img_contour_x = [img_contour_x;img_contour_x(1)];
    img_contour_y = [img_contour_y;img_contour_y(1)];
    %Display of the initial image
    img_buffer=m_nifti(:,:,param.slices(1)-1+i);
    imagesc(img_buffer')
    hold on 
    plot(img_contour_x,img_contour_y,'r',centerline(end,1,i),centerline(end,2,i),'xr','linewidth',1.2)
    plot(img_contour_x,img_contour_y,'r','markersize',8,'linewidth',1.2)    
    colormap gray, axis image;
    title('Segmentation de la coupe courante','FontWeight','bold','fontsize',16)
    axis([75 200 1 52])
    hold off
    axes(ha(2))
    img_buffer1=squeeze(m_nifti(:,ceil(end/2),:));
    imagesc(flipud(img_buffer1'))
    hold on
    colormap gray
    axis([1 320 50 250]);
    axis image
    po = plot([1 320],[320-param.slices(1)-1-i 320-param.slices(1)-1-i],'r','linewidth',1.1);
%     set(po,'linewidth',1.2)
%     img_buffer2 = squeeze(img_buffer1(:,param.slices(1)-1+i));
%     imagesc(img_buffer2)
    title('Position de la coupe','FontWeight','bold','fontsize',16)
    uicontrol('Style','text','units','normalized','Position', [0.2 0.24 4/30 0.03],'String','Surface de la coupe:','fontsize',ftsize,'BackgroundColor','w')
    uicontrol('Style','text','units','normalized','Position', [0.2+4/30 0.24 2/30 0.03],'String',area_per_slice(end,i),'fontsize',ftsize,'BackgroundColor','w')
	uicontrol('Style','text','units','normalized','Position', [0.2+6/30 0.24 2/30 0.03],'String','mm2','fontsize',ftsize,'BackgroundColor','w')
    uicontrol('Style','text','units','normalized','Position', [0.2 0.21 4/30 0.03],'String', '    Surface moyenne:','fontsize',ftsize,'BackgroundColor','w')
    uicontrol('Style','text','units','normalized','Position', [0.2+4/30 0.21 2/30 0.03],'String',average_area(end),'fontsize',ftsize,'BackgroundColor','w')
	uicontrol('Style','text','units','normalized','Position', [0.2+6/30 0.21 2/30 0.03],'String','mm2','fontsize',ftsize,'BackgroundColor','w')
    hold off
    g(i) = getframe(f);
    writeVideo(writerObj,g(i))
end

close(writerObj)
end
% movie(g,10)

% 
% figure('Renderer','zbuffer')
% Z = peaks;
% surf(Z); 
% axis tight
% set(gca,'NextPlot','replaceChildren');
% % Preallocate the struct array for the struct returned by getframe
% F(20) = struct('cdata',[],'colormap',[]);
% % Record the movie
% for j = 1:20 
%     surf(.01+sin(2*pi*j/20)*Z,Z)
%     F(j) = getframe;
% end
% movie(F,10)