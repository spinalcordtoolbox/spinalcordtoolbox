function scs_slider(nifti_img, m_center_line, contour, area, average_area)
% scs_slider 
%   Display a 3D matrix (ie. the image) with the contour of the segmented
%   ROI. The display starts at the middle slice (navigation is done
%   throughout the z axis) and at the first iteration.
%
%
% SYNTAX:
% [] = scs_slider(NIFTI_IMG, M_CENTER_LINE, CONTOUR)
%
% 
% INPUTS:
%
% M_RADIUS    
%   (number_of_slices x number_of_angles array) Radius smoothed
%
% M_CENTER_LINE_PREVIOUS
%   (Nx3 array) Coordinates of the spinal cord center line for each previous slice of M_VOLUME
%
% CONTOUR
%_________________________________________________________________________
% OUTPUTS:
% 
% NONE
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    scrsz = get(0,'ScreenSize'); 
    
    area(:,end+1)=area(:,end);
    area(:,end+1)=area(:,end);
    % Create a figure
    h = figure('Name','Segmentation visualization','position',[0 0 scrsz(3) scrsz(4)]);
    global h1 h2 ftsize
    
    %Initial value of the image at the center of the 3D matrix
    max_img=size(nifti_img,3);
    current_slice=round(max_img/2);
    max_iter=size(contour,4);
    current_iteration=max_iter;
    
    % find the pixel coordinate of the contour
    img_contour_x = contour(:,1,current_slice,current_iteration)+m_center_line(current_iteration,1,current_slice);
    img_contour_y = contour(:,2,current_slice,current_iteration)+m_center_line(current_iteration,2,current_slice);
    img_contour_x = [img_contour_x;img_contour_x(1)];
    img_contour_y = [img_contour_y;img_contour_y(1)];
    
    % Add a slider uicontrol to control the axial plane
    h1=uicontrol('Style', 'slider','Min',1,'Max',max_img,'Value',current_slice,'SliderStep',[1/(max_img-1) 1/(max_img-1)],...
        'units','normalized','Position', [0.7 0.25 0.2 0.04],'Callback', {@sliderZ_callback, nifti_img, m_center_line, contour, area}); 
    
    % Add a slider uicontrol to control the iteration
    h2=uicontrol('Style', 'slider','Min',1,'Max',max_iter,'Value',current_iteration,'SliderStep',[1/(max_iter-1) 1/(max_iter-1)],...
        'units','normalized','Position', [0.4 0.25 0.2 0.04],'Callback', {@sliderITER_callback, nifti_img, m_center_line, contour, area, average_area}); 
    
    %Display of the initial image
    img_buffer=nifti_img(:,:,current_slice);
    imagesc(img_buffer')
    hold on 
    plot(img_contour_x,img_contour_y,'r',m_center_line(current_iteration,1,current_slice),m_center_line(current_iteration,2,current_slice),'xr')
    %plot(img_contour_x,img_contour_y,'r-o','markersize',8)    
    colormap gray, axis image;
    hold off
    
    figure_size = get(h,'position');
    ftsize = mean(figure_size(3:4))/100;
    
    % Add a text uicontrol to label the Z slider.
	uicontrol('Style','text','units','normalized','Position', [0.7 0.30+0.03 6/30 0.03],'String','displayed slice','fontsize',ftsize)
    uicontrol('Style','text','units','normalized','Position', [0.7 0.30 2/30 0.03],'String',round(current_slice),'fontsize',0.8*ftsize)
    uicontrol('Style','text','units','normalized','Position', [0.7+2/30 0.30 2/30 0.03],'String','over','fontsize',0.8*ftsize)
    uicontrol('Style','text','units','normalized','Position', [0.7+4/30 0.30 2/30 0.03],'String',size(nifti_img,3),'fontsize',0.8*ftsize)
    
    % Add a text uicontrol to label the ITER slider.
    uicontrol('Style','text','units','normalized','Position', [0.4 0.30+0.03 6/30 0.03],'String','displayed iteration','fontsize',ftsize)
    uicontrol('Style','text','units','normalized','Position', [0.4 0.30 2/30 0.03],'String',round(current_iteration),'fontsize',0.8*ftsize)
    uicontrol('Style','text','units','normalized','Position', [0.4+2/30 0.30 2/30 0.03],'String','over','fontsize',0.8*ftsize)
    uicontrol('Style','text','units','normalized','Position', [0.4+4/30 0.30 2/30 0.03],'String',size(contour,4),'fontsize',0.8*ftsize)
    
    % Add a text uicontrol to show the area of the slice
	uicontrol('Style','text','units','normalized','Position', [0.7 0.24 2/30 0.03],'String','slice area:','fontsize',ftsize*0.8)
    uicontrol('Style','text','units','normalized','Position', [0.7+2/30 0.24 2/30 0.03],'String',area(current_iteration,current_slice),'fontsize',ftsize)
	uicontrol('Style','text','units','normalized','Position', [0.7+4/30 0.24 2/30 0.03],'String','mm2','fontsize',ftsize)
    
	% Add a text uicontrol to show the average area of this iteration
	uicontrol('Style','text','units','normalized','Position', [0.4 0.24 2/30 0.03],'String',{'Iteration' 'average area:'},'fontsize',ftsize*0.8)
    uicontrol('Style','text','units','normalized','Position', [0.4+2/30 0.24 2/30 0.03],'String',average_area(current_iteration),'fontsize',ftsize)
	uicontrol('Style','text','units','normalized','Position', [0.4+4/30 0.24 2/30 0.03],'String','mm2','fontsize',ftsize)
    
end

function sliderZ_callback(hObj,event, nifti_img, m_center_line, contour, area)
    global h2 ftsize
    % Called to set the current slice
    % when user moves the slider control 
    current_slice = round(get(hObj,'value'));
    current_iteration = round(get(h2,'value'));
    
    % find the pixel coordinate of the contour
    img_contour_x = contour(:,1,current_slice,round(current_iteration))+m_center_line(round(current_iteration),1,current_slice);
    img_contour_y = contour(:,2,current_slice,round(current_iteration))+m_center_line(round(current_iteration),2,current_slice);
    img_contour_x = [img_contour_x;img_contour_x(1)];
    img_contour_y = [img_contour_y;img_contour_y(1)];
    
    % Image display
    img_buffer=nifti_img(:,:,round(current_slice));
    imagesc(img_buffer')
    hold on   
    plot(img_contour_x,img_contour_y,'r', m_center_line(current_iteration,1,current_slice),m_center_line(current_iteration,2,current_slice),'xr')  
    colormap gray, axis image;
    hold off
    
    % Add a text uicontrol to label the Z slider.
    uicontrol('Style','text','units','normalized','Position', [0.7 0.30 2/30 0.03],'String',round(current_slice),'fontsize',0.8*ftsize)
    uicontrol('Style','text','units','normalized','Position', [0.7+4/30 0.30 2/30 0.03],'String',size(nifti_img,3),'fontsize',0.8*ftsize)
    
    % Add a text uicontrol to show the area of the slice
    uicontrol('Style','text','units','normalized','Position', [0.7+2/30 0.24 2/30 0.03],'String',area(round(current_iteration),round(current_slice)),'fontsize',ftsize)
end

function sliderITER_callback(hObj,event, nifti_img, m_center_line, contour, area, average_area)
    global h1 ftsize
    % Called to set the current slice
    % when user moves the slider control 
    current_iteration = round(get(hObj,'value'));
    current_slice = round(get(h1,'value'));
    
    % find the pixel coordinate of the contour
    img_contour_x = contour(:,1,current_slice,round(current_iteration))+m_center_line(round(current_iteration),1,current_slice);
    img_contour_y = contour(:,2,current_slice,round(current_iteration))+m_center_line(round(current_iteration),2,current_slice);
    img_contour_x = [img_contour_x;img_contour_x(1)];
    img_contour_y = [img_contour_y;img_contour_y(1)];
    
    % Image display
    img_buffer=nifti_img(:,:,round(current_slice));
    imagesc(img_buffer')
    hold on  
    plot(img_contour_x,img_contour_y,'r', m_center_line(current_iteration,1,current_slice),m_center_line(current_iteration,2,current_slice),'xr')  
    colormap gray, axis image;
    hold off
    
    % Add a text uicontrol to label the ITER slider.
    uicontrol('Style','text','units','normalized','Position', [0.4 0.30 2/30 0.03],'String',round(current_iteration),'fontsize',0.8*ftsize)
    uicontrol('Style','text','units','normalized','Position', [0.4+4/30 0.30 2/30 0.03],'String',size(contour,4),'fontsize',0.8*ftsize)
    
	% Add a text uicontrol to show the area of the slice
    uicontrol('Style','text','units','normalized','Position', [0.7+2/30 0.24 2/30 0.03],'String',area(round(current_iteration),round(current_slice)),'fontsize',ftsize)
    
    % Add a text uicontrol to show the average area of this iteration
    uicontrol('Style','text','units','normalized','Position', [0.4+2/30 0.24 2/30 0.03],'String',average_area(round(current_iteration)),'fontsize',ftsize)
end