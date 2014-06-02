function varargout = j_mri_view(varargin)
% GUI_JULIEN M-file for j_mri_view.fig
%      GUI_JULIEN, by itself, creates a new GUI_JULIEN or raises the existing
%      singleton*.
%
%      H = GUI_JULIEN returns the handle to a new GUI_JULIEN or the handle to
%      the existing singleton*.
%
%      GUI_JULIEN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI_JULIEN.M with the given input arguments.
%
%      GUI_JULIEN('Property','Value',...) creates a new GUI_JULIEN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before j_mri_view_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to j_mri_view_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Copyright 2002-2003 The MathWorks, Inc.

% Edit the above text to modify the response to help j_mri_view

% Last Modified by GUIDE v2.5 30-Oct-2007 15:21:57

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @j_mri_view_OpeningFcn, ...
                   'gui_OutputFcn',  @j_mri_view_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before j_mri_view is made visible.
function j_mri_view_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to j_mri_view (see VARARGIN)

% Choose default command line output for j_mri_view
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes j_mri_view wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = j_mri_view_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;









% --- Executes on button press in pushbutton1.
function  pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% initialization
index_fig = 100;

% load fMRI data
opt.windows_title		= 'Please select MR images(s)';
opt.file_selection		= 'spm';
opt.format				= 'nii';
[data hdr] = j_mri_read('',opt);
if ~data, return; end

% display temporal time series
j_fn_4Dview(index_fig,data,'plot');
set(index_fig,'Position',[20 20 560 420]); % set the position to bottom left
index_fig = index_fig + 1;

% display spatial time series
j_fn_4Dview(index_fig,data,'display','HOLA');
index_fig = index_fig + 1;
colormap gray



% data = j_mri_read('D:\data_irm\analyses_irm\2007-10-23_cardiac-gating_T1-correction\test\');
% [nx ny nz nt] = size(data);
% 
% fHandle = figure('HandleVisibility','on','IntegerHandle','off','Visible','off');
% % aHandle = axes('Parent',fHandle);
% % pHandles = plot(PlotData,'Parent',aHandle);
% set(fHandle,'Visible','on');
% opt.h_fig = fHandle;
% j_displayMRI(data(:,:,:,1),opt);


% save handles
% handles.fHandle = fHandle;
handles.index_fig = index_fig;
guidata(handles.figure1,handles)








% =========================================================================
% Add fMRI series
% =========================================================================
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% retrieve handles
index_fig = handles.index_fig;

% load fMRI data
opt.file_selection = 'spm';
[data hdr] = j_mri_read('',opt);
% if ~fname_data, return; end

% display spatial time series
j_fn_4Dview(index_fig,data,'display','TODO');
index_fig = index_fig + 1;
colormap gray

% update handles
handles.index_fig = index_fig;
guidata(handles.figure1,handles)






% =========================================================================
% Close all
% =========================================================================
% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% retrieve handles
index_fig = handles.index_fig;

for i=100:index_fig-1
    if ishandle(i), close(i); end
end

