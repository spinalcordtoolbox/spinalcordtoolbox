function scs_get_slice(varargin)
% scs_slider_3dmatrix
%	creates a figure for the visualization of a 3D matrix. 
%   navigation is done throughout the z axis
%
% SYNTAX:
% [] = scs_slider_3dmatrix(VARARGIN)
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
    if nargin==2, action_title = varargin{2}; elseif nargin==1, action_title = 'Set Slice'; end
    title = 'Get Slice number';
    % Create a figure
    scrsz = get(0,'ScreenSize'); 
    h=figure('Name',title,'position',[0 0 scrsz(3) scrsz(4)]);
    
    %Initial value of the image at the center of the 3D matrix
    max_img=size(nifti_img,3);
    current_slice=round(max_img/2);
    global Slice;
    Slice = current_slice;
    
    % Add a slider uicontrol to control the vertical scaling of the
    % surface object. Position it under the Clear button.
    uicontrol('Style', 'slider','Min',1,'Max',max_img,'Value',current_slice,'SliderStep',[1/(max_img-1) 1/(max_img-1)],...
        'units','normalized','Position', [0.7 0.25 0.2 0.04],'Callback', {@sliderZ_callback, nifti_img}); 
    
    % Add a text to choose this slice number
    uicontrol('Style','text','units','normalized','Position', [0.7 0.52 0.2 0.03],'FontWeight','bold','FontSize',15,'ForegroundColor',[0.8 0 0],'String',action_title)
    uicontrol('Style','text','units','normalized','Position', [0.7 0.50 0.2 0.03],'FontWeight','bold','FontSize',15,'String','Press any Button to choose current slice')
    
    %Display of the initial image
    img_buffer=nifti_img(:,:,current_slice);
    imagesc(img_buffer)
    colormap gray, axis image;
    
    % Add a text uicontrol to label the slider.
    uicontrol('Style','text','units','normalized','Position', [0.7 0.30 2/30 0.03],'String',round(current_slice))
    uicontrol('Style','text','units','normalized','Position', [0.7+2/30 0.30 2/30 0.03],'String','over')
    uicontrol('Style','text','units','normalized','Position', [0.7+4/30 0.30 2/30 0.03],'String',size(nifti_img,3))
    
    pause;
end

function sliderZ_callback(hObj,event, nifti_img)
    % Called to set the current slice
    % when user moves the slider control 
    current_slice = get(hObj,'value');
    setappdata(0, 'current_slice', current_slice);
    % Image display
    img_buffer=nifti_img(:,:,round(current_slice));
    imagesc(img_buffer)
    colormap gray, axis image;
    
    global Slice;
    Slice = current_slice;
    
    % Slider label
    uicontrol('Style','text','units','normalized','Position', [0.7 0.30 2/30 0.03],'String',round(current_slice))
    uicontrol('Style','text','units','normalized','Position', [0.7+2/30 0.30 2/30 0.03],'String','over')
    uicontrol('Style','text','units','normalized','Position', [0.7+4/30 0.30 2/30 0.03],'String',size(nifti_img,3))
end
