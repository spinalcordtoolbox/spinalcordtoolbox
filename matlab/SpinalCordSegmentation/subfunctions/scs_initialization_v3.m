function [m_nifti,m_volume,dims,scales, nom_radius, m_center_line, m_tangent_cord, param] = scs_initialization(input, param)
% scs_initialization 
%	reads the header to find the current reference and reorient in PSR
%   reads and stores raw NIfTI images in MATLAB matrices
%	smooths the image (optional)
%	resamples the image (optional)
%   sets the nominal radius in pixels
%   asks the user to enter the center line
%   interpolates the user data set
%	computes the initial unit tengent vector
%
% SYNTAX:
% [M_NIFTI,M_VOLUME,DIMS,SCALES, NOM_RADIUS, M_CENTER_LINE, M_TANGENT_CORD, PARAM] = scs_initialization(INPUT, PARAM)
%
% _________________________________________________________________________
% INPUTS:
%
% INPUT     
%   (string) a file name of an image (supported format see IMREAD).
%
% PARAM
%   Struct containing an assortment of user defined parameters (see user
%   guide for details)
% _________________________________________________________________________
% OUTPUTS:
%
% M_NIFTI
%   (XxYxZ array) Voxels intensity of raw NIfTI image
%
% M_VOLUME
%   (XxYxZ array) Voxels intensity of desired NIfTI image (FOV)
%
% DIMS
%   (4x1 array) Dimensions of M_VOLUME
%
% SCALES
%   (4x1 array) Scales in mm of X, Y and Z axes in M_VOLUME
%
% M_CENTER_LINE
%   (Nx3 array) Coordinates of the spinal cord for each slice of M_VOLUME
%
% M_TANGENT_CORD
%   Coordinates of the unit tangent vector
%
% PARAM
%   Struct containing an assortment of user defined parameters (see user
%   guide for details)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fsloutput = ['export FSLOUTPUTTYPE=NIFTI;'];


if strcmp(input(end-3:end),'.mat') == 0
    
    param.nii=load_nii(input);
    m_nifti=double(param.nii.img); 
    dims=size(m_nifti); scales=param.nii.scales;
    
else
    load(input);
    m_nifti=m_phantom;
    scales=[1 1 1 1];
    dims=[size(m_nifti)'; 1]; dims=dims(:)';
    param.file_type = '.mat';
    param.orient_init = 'PSL';
    param.orient_new = param.orient_init;
end


% Supression of undesired slices
% Ask user for spinal cord bounds
if size(param.centerline) == [1, 1]
    
    global interval;
    global is_closed; is_closed = 0;
    scs_get_slice_v2(m_nifti);
    while ~is_closed, pause(1); end
    param.slices(2) = max(interval);
    param.slices(1) = min(interval);
    clear global interval; clear global is_closed;
    
    m_volume=m_nifti;
    m_volume(:,:,param.slices(2)+1:dims(3))=[];   % Suppression of the last slices
    m_volume(:,:,1:param.slices(1))=[];         % Suppression of the first slices

end

% Oversampling of the volume of interest
m_volume = imresize(m_volume,param.resampling); 

% Smoothing of the slices
gaussian_mask = [1 2 1; 2 4 2; 1 2 1];
% gaussian_mask = [1 2 4 2 1; 2 4 8 4 2; 4 8 32 8 4; 2 4 8 4 2; 1 2 4 2 1];
gaussian_mask = gaussian_mask/sum(sum(gaussian_mask));
m_volume_smoothed = zeros(size(m_volume));
for i=1:size(m_volume,3)
    m_volume_smoothed(:,:,i) = conv2(m_volume(:,:,i),gaussian_mask,'same');
end
m_volume=m_volume_smoothed;
clear m_volume_smoothed

scales=[scales(1)/param.resampling scales(2)/param.resampling scales(3) scales(4)];

% Set the nominal radius at 4 mm in pixel
nom_radius = param.nom_radius/min([scales(1) scales(2)]);

% Definition of the initial centerline 
if size(param.centerline) == [1, 1]     % If no centerline is given, asks the user to give one
    info{1}='no';
    while strcmp(info{1},'no')
        % Display of the different slices with a step corresponding to
        % param.interval and selection of the center of the spinal cord
        clear x y z
        scrsz = get(0,'ScreenSize'); % full screen
        no_image=1;
        intervals = 1:round(param.interval/scales(3)):size(m_volume,3);
        for i=intervals
            
            img_buffer=m_volume(:,:,i);
            close all;
            f = figure(i+param.slices(1)-1);
            set(f,'Position',[1 1 scrsz(3) scrsz(4)]), imagesc(img_buffer'), colormap gray, axis image
            ylabel('y')
            xlabel('x')
            % --------------------
            %title('Spinal Cord Center Line Initialization')
            % Display text around the image
            ylim=get(gca,'YLim');xlim=get(gca,'XLim'); ftsize = mean(get(f,'position'))/50;
            text(xlim(1),ylim(1), 'Click on the center of the spinal cord', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[0.8 0 0], 'Fontsize',ftsize)
            text(xlim(2),ylim(2)*1.15, ['Still ' num2str(size(intervals,2)+1-find(intervals==i)) ' remaining'], 'VerticalAlignment','bottom','HorizontalAlignment','right','Fontsize',ftsize)
            
          
            
            % Display text around the orientation (PSR)
            text(xlim(1),(ylim(1)+ylim(2))/2, 'L', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
            text(xlim(2),(ylim(1)+ylim(2))/2, 'R', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',ftsize)
            text((xlim(1)+xlim(2))/2,ylim(1), 'P', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
            text((xlim(1)+xlim(2))/2,ylim(2), 'A', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
            % --------------------
            
            [x(no_image) y(no_image)] = ginput(1);
            z(no_image)=i;
%             if x(no_image)
            close(i+param.slices(1)-1); clear img_buffer;
            no_image=no_image+1;
            
        end
        % For the last slice
        if i~=size(m_volume,3)
            i=size(m_volume,3);
            img_buffer=m_volume(:,:,size(m_volume,3));
            close all;
            f=figure(size(m_volume,3)+param.slices(1)-1);
            set(f,'Position',[1 1 scrsz(3) scrsz(4)]), imagesc(img_buffer'), colormap gray, axis image
            ylabel('y')
            xlabel('x')
            %title('Spinal Cord Center Line Initialization')
            ylim=get(gca,'YLim');xlim=get(gca,'XLim'); ftsize = mean(get(f,'position'))/50;
            text(xlim(1),ylim(1), 'Click on the center of the spinal cord', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[0.8 0 0], 'Fontsize',ftsize)
            text(xlim(2),ylim(2)*1.15, ['This is the last one !'], 'VerticalAlignment','bottom','HorizontalAlignment','right','Fontsize',ftsize)
            text(xlim(1),(ylim(1)+ylim(2))/2, 'L', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
            text(xlim(2),(ylim(1)+ylim(2))/2, 'R', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',ftsize)
            text((xlim(1)+xlim(2))/2,ylim(1), 'P', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
            text((xlim(1)+xlim(2))/2,ylim(2), 'A', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
            [x(no_image) y(no_image)] = ginput(1);
            z(no_image)=i;
            close(size(m_volume,3)+param.slices(1)-1);
        end   
    

        % Cubic spline in X and Y
        x = interp1(z, x, 1:size(m_volume,3),'spline');
        y = interp1(z, y, 1:size(m_volume,3),'spline');
        z = 1:size(m_volume,3);
        m_center_line=[x' y' z'];

        % Display of the initial center_line
        f=figure('color','w');  
        set(f,'Position',[1 1 scrsz(3) scrsz(4)]);
        title('Click to exit the view!','FontSize',18)
        % xz slice at the middle of the spline    
        subplot(2,3,[4,5])
        img_buffer=squeeze(m_volume(:,round(mean(m_center_line(:,2))),:));
        imagesc(img_buffer'); colormap gray, axis image
        hold on
        plot(m_center_line(:,1),m_center_line(:,3),'r')
        title('sagittal view')
        ylim=get(gca,'YLim');xlim=get(gca,'XLim'); ftsize = mean(get(f,'position'))/50;
        text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
        text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',ftsize)
        text((xlim(1)+xlim(2))/2,ylim(1), 'S', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
        text((xlim(1)+xlim(2))/2,ylim(2), 'I', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
        xlabel('x')
        ylabel('z')
        hold off
        % yz slice 
        subplot(2,3,[3,6])
        img_buffer=squeeze(m_volume(round(mean(m_center_line(:,1))),:,:));
        imagesc(img_buffer'); colormap gray, axis image
        hold on
        plot(m_center_line(:,2),m_center_line(:,3),'r')
        title('coronal view')
        ylim=get(gca,'YLim');xlim=get(gca,'XLim'); ftsize = mean(get(f,'position'))/50;
        text(xlim(1),(ylim(1)+ylim(2))/2, 'S', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
        text(xlim(2),(ylim(1)+ylim(2))/2, 'I', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',ftsize)
        text((xlim(1)+xlim(2))/2,ylim(1), 'L', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
        text((xlim(1)+xlim(2))/2,ylim(2), 'R', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
        xlabel('y')
        ylabel('z')
        hold off
        % xy slice 
        subplot(2,3,[1,2])
        img_buffer=squeeze(m_volume(:,:,round(mean(m_center_line(:,3)))));
        imagesc(img_buffer'); colormap gray, axis image
        hold on
        plot(m_center_line(:,1),m_center_line(:,2),'r')
        title('axial view')
        ylim=get(gca,'YLim');xlim=get(gca,'XLim'); ftsize = mean(get(f,'position'))/50;
        text(xlim(1),(ylim(1)+ylim(2))/2, 'L', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
        text(xlim(2),(ylim(1)+ylim(2))/2, 'R', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',ftsize)
        text((xlim(1)+xlim(2))/2,ylim(1), 'P', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
        text((xlim(1)+xlim(2))/2,ylim(2), 'A', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',ftsize)
        xlabel('x')
        ylabel('y')
        hold off

        % Exit the initial view
        clear info
        info{1}=questdlg('Are you satisfied with the initialization?','Confirmation','yes','no','yes');
        if isempty(info{1})
            info{1}='yes';
        end
        delete(f)
    end
else    % If a centerline is given
    x = param.centerline(1,:);
    y = param.centerline(2,:);
    z = param.centerline(3,:);
    % Cubic spline in X and Y
    x = interp1(z, x, 1:size(m_volume,3),'spline');
    y = interp1(z, y, 1:size(m_volume,3),'spline');
    z = 1:size(m_volume,3);
    m_center_line=[x' y' z'];
end

close all
pause(0.1);

% Verification that the centerline is within the volume
if min(min(m_center_line)) < 1 % no negative coordinates
    scs_warning_error(219, param)
elseif max(m_center_line(:,1)) > size(m_volume,1) || max(m_center_line(:,2)) > size(m_volume,2) || max(m_center_line(:,3)) > size(m_volume,3)   % no coordinates out of the volume
	scs_warning_error(219, param)
end

% Computation of the unit vector tangent
%   Computation of the derivatives
dx=x(2:size(x,2))-x(1:size(x,2)-1);
dy=y(2:size(y,2))-y(1:size(y,2)-1);
dz=ones(size(dx,2),1)*scales(3);
m_tangent_cord = [dx' dy' dz];
%   Convert in mm
m_tangent_cord(:,1)=m_tangent_cord(:,1)*scales(1);
m_tangent_cord(:,2)=m_tangent_cord(:,2)*scales(2);
%   Computation of the norm
norm_tangent=sqrt(sum(m_tangent_cord.*m_tangent_cord, 2));  %euclidian norm
%   Computation of the unit vector tangent
m_tangent_cord(:,1)=m_tangent_cord(:,1)./norm_tangent;
m_tangent_cord(:,2)=m_tangent_cord(:,2)./norm_tangent;
m_tangent_cord(:,3)=m_tangent_cord(:,3)./norm_tangent;
%   Convert in pixels
m_tangent_cord(:,1)=m_tangent_cord(:,1)/scales(1);
m_tangent_cord(:,2)=m_tangent_cord(:,2)/scales(2);

% Transpose matrix (modification par J et Y)
m_tangent_cord = m_tangent_cord';
m_tangent_cord = [m_tangent_cord m_tangent_cord(:,end)]; %Add another tangent to have the same length as m_center_line
m_center_line = m_center_line';

% Padding of the m_volume matrix
m_volume=m_nifti;
m_volume=imresize(m_volume,param.resampling);
m_volume_smoothed = zeros(size(m_volume));
for i=1:size(m_volume,3)
    m_volume_smoothed(:,:,i) = conv2(m_volume(:,:,i),gaussian_mask,'same');
end
m_volume=m_volume_smoothed;
clear m_volume_smoothed
% step=ceil(nom_radius+2*scales(3));
step=0;
m_volume(:,:,param.slices(2)+1+step:dims(3))=[];    % Suppression of the last slices
m_volume(:,:,1:param.slices(1)-step)=[];            % Suppression of the first slices

end % END OF SCS_INITIALIZATION
