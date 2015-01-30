function scs_get_slice_v2(varargin)
% scs_get_slice_v2
%	gui that selects the field of view (ie. top and bottom slices) 
%   for further computation
%
% SYNTAX:
% [] = scs_get_slice_v2(VARARGIN)
%
% _________________________________________________________________________
% INPUTS:
%
% M_NIFTI
%   (XxYxZ array) Voxels intensity of raw NIfTI image
%
% TITLE (optional)
%	(string) Title of the action
% _________________________________________________________________________
% OUTPUTS:
%
% NONE
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nifti_img = varargin{1};
	% Set default title
    if nargin==2, action_title = varargin{2}; elseif nargin==1, action_title = 'Selection of the field of view'; end
    title = 'Get Slice number';
    % Create a figure
    scrsz = get(0,'ScreenSize'); 
    h=figure('Name',title,'position',[0 0 scrsz(3) scrsz(4)]);

    %Initial value of the image at the center of the 3D matrix
    max_img=size(nifti_img,3);
    current_slice=round(max_img/2);
    global Slice;
    Slice = current_slice;
    global interval;    interval = [current_slice current_slice];
    global is_top; is_top = 0;
    global is_bottom; is_bottom = 0;
    
    
    %Display sagittal view
    hax1 = subplot(3,3,[1 4 7]);
    img_buffer=squeeze(nifti_img(round(end/2),:,:));
    imagesc(img_buffer'); colormap gray, axis image;
    set(gca,'YDir','normal')
    set(hax1,'XTick',[]);
    ylim=get(gca,'YLim');xlim=get(gca,'XLim');
    text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(1), 'S', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(2), 'I', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    hold on
    plot([1 size(img_buffer',1)],[round(current_slice) round(current_slice)],'r')
    hold off
    
    % Add a slider uicontrol to control the vertical scaling of the
    % surface object. Position it under the Clear button.
    uicontrol('Style', 'slider','Min',1,'Max',max_img,'Value',current_slice,'SliderStep',[1/(max_img-1) 1/(max_img-1)],...
        'units','normalized','Position', [0.7 0.17 0.2 0.04],'Callback', {@sliderZ_callback, nifti_img}); 
    
    % Add a text to choose this slice number
    uicontrol('Style','text','units','normalized','Position', [0.7 0.28 0.2 0.03],'FontWeight','bold','FontSize',15,'String',action_title)
    uicontrol('Style','text','units','normalized','Position', [0.7 0.25 0.2 0.04],'FontSize',14,'String','Use the slider to navigate. Press OK to select. Press CONFIRM to finish.')
    
    %Display of the initial image
    hax2 = subplot(3,3,[2 3 5 6]);
    img_buffer=nifti_img(:,:,current_slice);
    imagesc(img_buffer')
    colormap gray, axis image;
    set(gca,'YDir','normal')
    set(hax2,'XTick',[],'YTick',[]);
    ylim=get(gca,'YLim');xlim=get(gca,'XLim');
    text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(1), 'L', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(2), 'R', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)

    
    % Add a text uicontrol to label the slider.
    uicontrol('Style','text','units','normalized','Position', [0.7 0.22 2/30 0.03],'String',round(current_slice))
    uicontrol('Style','text','units','normalized','Position', [0.7+2/30 0.22 2/30 0.03],'String','over')
    uicontrol('Style','text','units','normalized','Position', [0.7+4/30 0.22 2/30 0.03],'String',size(nifti_img,3))
    
    % Add a text uicontrol to indicate the top/bottom slice.
    uicontrol('Style','text','units','normalized','Position', [0.45 0.42 2/30 0.03],'String','Top slice','FontSize',15,'FontWeight','bold')
    uicontrol('Style','text','units','normalized','Position', [0.45+2/30 0.42 2/30 0.03],'FontSize',15,'String',round(interval(1)))
    
    uicontrol('Style','text','units','normalized','Position', [0.45 0.38 2/30 0.03],'String','Bottom slice','FontSize',15,'FontWeight','bold')
    uicontrol('Style','text','units','normalized','Position', [0.45+2/30 0.38 2/30 0.03],'FontSize',15,'String',round(interval(2)))
    
    % Add confirmation button
    % OK : select the top/bottom slice
    uicontrol('Style','pushbutton','units','normalized','Position', [0.45+4/30 0.42 2/30 0.03],'string','OK','Callback',{@topOK_callback, nifti_img});
    uicontrol('Style','pushbutton','units','normalized','Position', [0.45+4/30 0.38 2/30 0.03],'string','OK','Callback',{@bottomOK_callback, nifti_img});
    % CANCEL : remove the selected slice
    uicontrol('Style','pushbutton','units','normalized','Position', [0.45+6/30 0.42 2/30 0.03],'string','CANCEL','Callback',{@topCANCEL_callback, nifti_img});
    uicontrol('Style','pushbutton','units','normalized','Position', [0.45+6/30 0.38 2/30 0.03],'string','CANCEL','Callback',{@bottomCANCEL_callback, nifti_img});
    % CONFIRM : exit the figure
    uicontrol(h,'Style','pushbutton','units','normalized','Position', [0.45+10/30 0.40 2/30 0.03],'string','CONFIRM','FontSize',15,'Callback',{@confirm_callback});

    
    %pause;
end

function sliderZ_callback(hObj,event, nifti_img)
    % Called to set the current slice
    % when user moves the slider control
    global is_top; global is_bottom; global interval;
    current_slice = get(hObj,'value');
    setappdata(0, 'current_slice', current_slice);
    % Image display
    hax2 = subplot(3,3,[2 3 5 6]);
    img_buffer=nifti_img(:,:,round(current_slice));
    imagesc(img_buffer')
    colormap gray, axis image;     set(gca,'YDir','normal');
    set(hax2,'XTick',[],'YTick',[]);
    ylim=get(gca,'YLim');xlim=get(gca,'XLim'); 
    text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(1), 'L', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(2), 'R', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)

    % Plotting current position
    hax1 = subplot(3,3,[1 4 7]);
    img_buffer=squeeze(nifti_img(round(end/2),:,:));
    imagesc(img_buffer'); colormap gray, axis image; set(gca,'YDir','normal'); set(hax1,'XTick',[]);
    ylim=get(gca,'YLim');xlim=get(gca,'XLim');
    text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(1), 'S', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(2), 'I', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    hold on
    plot([1 size(img_buffer',1)],[round(current_slice) round(current_slice)],'r')
    if is_bottom, plot([1 size(img_buffer',1)],[round(interval(2)) round(interval(2))],'g'); end
    if is_top, plot([1 size(img_buffer',1)],[round(interval(1)) round(interval(1))],'g'); end
    hold off
    
    global Slice;
    Slice = current_slice;
    
    % Slider label
    uicontrol('Style','text','units','normalized','Position', [0.7 0.22 2/30 0.03],'String',round(current_slice))
    uicontrol('Style','text','units','normalized','Position', [0.7+2/30 0.22 2/30 0.03],'String','over')
    uicontrol('Style','text','units','normalized','Position', [0.7+4/30 0.22 2/30 0.03],'String',size(nifti_img,3))
end

function topOK_callback(hObj,event,nifti_img)

    global Slice;
    global interval;
    interval(1) = round(Slice);
    global is_top; is_top = 1;
    global is_bottom;
    
    uicontrol('Style','text','units','normalized','Position', [0.45 0.42 2/30 0.03],'String','Top slice','FontSize',15,'FontWeight','bold')
    uicontrol('Style','text','units','normalized','Position', [0.45+2/30 0.42 2/30 0.03],'FontSize',15,'String',round(interval(1)),'ForegroundColor',[0 0.8 0])

    % Plot the selected top position
    hax1 = subplot(3,3,[1 4 7]);
    img_buffer=squeeze(nifti_img(round(end/2),:,:));
    imagesc(img_buffer'); colormap gray, axis image;    set(gca,'YDir','normal'); set(hax1,'XTick',[]);
    ylim=get(gca,'YLim');xlim=get(gca,'XLim');
    text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(1), 'S', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(2), 'I', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    hold on
    if is_bottom, plot([1 size(img_buffer',1)],[round(interval(2)) round(interval(2))],'g'); end
    plot([1 size(img_buffer',1)],[round(interval(1)) round(interval(1))],'g')
    hold off
    
end

function bottomOK_callback(hObj,event,nifti_img)

    global Slice;
    global interval;
    interval(2) = round(Slice);
    global is_bottom; is_bottom = 1;
    global is_top;
    
    uicontrol('Style','text','units','normalized','Position', [0.45 0.38 2/30 0.03],'String','Bottom slice','FontSize',15,'FontWeight','bold')
    uicontrol('Style','text','units','normalized','Position', [0.45+2/30 0.38 2/30 0.03],'FontSize',15,'String',round(interval(2)),'ForegroundColor',[0 0.8 0])
    
    % Plot the selected bottom position
    hax1 = subplot(3,3,[1 4 7]);
    img_buffer=squeeze(nifti_img(round(end/2),:,:));
    imagesc(img_buffer'); colormap gray, axis image;     set(gca,'YDir','normal'); set(hax1,'XTick',[]);
    ylim=get(gca,'YLim');xlim=get(gca,'XLim');
    text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(1), 'S', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(2), 'I', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)

    hold on
    if is_top, plot([1 size(img_buffer',1)],[round(interval(1)) round(interval(1))],'g'); end
    plot([1 size(img_buffer',1)],[round(interval(2)) round(interval(2))],'g')
    hold off
    
end

function topCANCEL_callback(hObj,event,nifti_img)

    global Slice;
    global interval;
    interval(1) = round(Slice);
    global is_top; is_top = 0;
    global is_bottom;
    
    uicontrol('Style','text','units','normalized','Position', [0.45 0.42 2/30 0.03],'String','Top slice','FontSize',15,'FontWeight','bold')
    uicontrol('Style','text','units','normalized','Position', [0.45+2/30 0.42 2/30 0.03],'FontSize',15,'String',round(interval(1)))

    % Erase the selected top position
    hax1 = subplot(3,3,[1 4 7]);
    img_buffer=squeeze(nifti_img(round(end/2),:,:));
    imagesc(img_buffer'); colormap gray, axis image; set(gca,'YDir','normal'); set(hax1,'XTick',[]);
    ylim=get(gca,'YLim');xlim=get(gca,'XLim');
    text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(1), 'S', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(2), 'I', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)

    hold on
    if is_bottom, plot([1 size(img_buffer',1)],[round(interval(2)) round(interval(2))],'g'); end
    plot([1 size(img_buffer',1)],[round(Slice) round(Slice)],'r')
    hold off
    
end

function bottomCANCEL_callback(hObj,event,nifti_img)

    global Slice;
    global interval;
    interval(2) = round(Slice);
    global is_bottom; is_bottom = 0;
    global is_top;
    
    uicontrol('Style','text','units','normalized','Position', [0.45 0.38 2/30 0.03],'String','Bottom slice','FontSize',15,'FontWeight','bold')
    uicontrol('Style','text','units','normalized','Position', [0.45+2/30 0.38 2/30 0.03],'FontSize',15,'String',round(interval(2)))
    
    % Erase the selected bottom position
    hax1 = subplot(3,3,[1 4 7]);
    img_buffer=squeeze(nifti_img(round(end/2),:,:));
    imagesc(img_buffer'); colormap gray, axis image; set(gca,'YDir','normal'); set(hax1,'XTick',[]);
    ylim=get(gca,'YLim');xlim=get(gca,'XLim');
    text(xlim(1),(ylim(1)+ylim(2))/2, 'P', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text(xlim(2),(ylim(1)+ylim(2))/2, 'A', 'VerticalAlignment','bottom','HorizontalAlignment','right','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(1), 'S', 'VerticalAlignment','top','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)
    text((xlim(1)+xlim(2))/2,ylim(2), 'I', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[1 1 1], 'Fontsize',14)

    hold on
    if is_top, plot([1 size(img_buffer',1)],[round(interval(1)) round(interval(1))],'g'); end
    plot([1 size(img_buffer',1)],[round(Slice) round(Slice)],'r')
    hold off
    
end

function confirm_callback(hObj,event,handles)

    global interval;
    global is_bottom; global is_top;
    global is_closed;
    
    if ~is_bottom || ~is_top
        message = sprintf('You haven''t select any slice!');
        warndlg(message);
    elseif interval(1) == interval(2)
        message = sprintf('Incorrect selection !');
        warndlg(message);
    else
        info{1}=questdlg('Are you satisfied with the initialization?','Confirmation','yes','no','yes');
        if isempty(info{1}), info{1}='yes'; end
        if strcmp(info{1},'yes')
            is_closed = 1;
            delete(gcf)
        end
    end
end
