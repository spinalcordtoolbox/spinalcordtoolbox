 function m_center_line = sct_get_centerline(param)
% INPUT :
%     param.interval;
%     param.img; 3D matrix
% OUTPUT :
%     N by 3 matrix.
%     N : nb of slices
%     columns 1 & 2 : coordinates of the slice
%     column 3 : slice number

if ~isfield(param,'close'), param.close = 1; end
m_volume_raw = read_avw(param.img);
if ~isfield(param,'slices'), param.slices = [0 size(m_volume_raw,3)-1]; end

%choose slices
m_volume=m_volume_raw(:,:,(param.slices(1)+1):(param.slices(2)+1));

info{1}='no';
    while strcmp(info{1},'no')
        % Display of the different slices with a step corresponding to
        % param.interval and selection of the center of the spinal cord
        scrsz = get(0,'ScreenSize'); % full screen
        no_image=1;
        intervals = 1:round(param.interval):size(m_volume,3);
        for i=intervals
            
            img_buffer=m_volume(:,:,i);
            if param.close, close all; end
            f = figure(i+1-1);
            set(f,'Position',[1 1 scrsz(3) scrsz(4)]), imagesc(img_buffer'), colormap jet, axis normal
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
            close(i+1-1); clear img_buffer;
            no_image=no_image+1;
            
        end
        % For the last slice
        if i~=size(m_volume,3)
            i=size(m_volume,3);
            img_buffer=m_volume(:,:,size(m_volume,3));
            close all;
            f=figure(size(m_volume,3)+1-1);
            set(f,'Position',[1 1 scrsz(3) scrsz(4)]), imagesc(img_buffer'), colormap gray, axis normal
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
            close(size(m_volume,3)+1-1);
        end   
    

        % Cubic spline in X and Y
        if length(z)>1
            x = interp1(z, x, 1:size(m_volume,3),'spline');
            y = interp1(z, y, 1:size(m_volume,3),'spline');
            z = 1:size(m_volume,3);
        end
        m_center_line=[x' y(end:-1:1)' z'];

        % Display of the initial center_line
        f=figure('color','w');  
        set(f,'Position',[1 1 scrsz(3) scrsz(4)]);
        title('Click to exit the view!','FontSize',18)
        % xz slice at the middle of the spline    
        subplot(2,3,[4,5])
        img_buffer=squeeze(m_volume(:,round(mean(m_center_line(:,2))),:));
        imagesc(img_buffer'); colormap gray, axis normal
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
        imagesc(img_buffer'); colormap gray, axis normal
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
        imagesc(img_buffer'); colormap gray, axis normal
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
end