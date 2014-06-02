function [filenames, pathname] = j_uigetfiles(opt_getfiles)
% This is a Java interfaced version of UIGETFILES, that brings multiple file
% open dialog box. 
%
% [filenames, pathname] = uigetfiles; displays a dialog box file browser
% from which the user can select multiple files.  The selected files are
% returned to FILENAMES as an arrayed strings. The directory containing
% these files is returned to PATHNAME as a string. 
%
% A successful return occurs only if the files exist.  If the user selects
% a  file that does not exist, an error message is displayed to the command
% line.  If the Cancel button is selected, zero is assigned to FILENAMES
% and current directory is assigned to PATHNAME. 
% 
% This program has an equivalent function to that of a C version of
% "uigetfiles.dll" downloadable from www.mathworks.com under support, file
% exchange (ID: 331). 
%
% It should work for matlab with Java 1.3.1 or newer.
%
% INPUT
% file_filter           'img', 'image'
%
% Shanrong Zhang
% Department of Radiology
% University of Texas Southwestern Medical Center
% 02/09/2004
% email: shanrong.zhang@utsouthwestern.edu
%
% Modified by Julien Cohen-Adad for allowing file filter
% 2007-04-04

% default initialization
if ~exist('opt_getfiles'), opt_getfiles = []; end
file_filter = '';
windows_title = 'Please select file(s)';

% user initialization
if isfield(opt_getfiles,'file_filter'), file_filter = opt_getfiles.file_filter; end
if isfield(opt_getfiles,'windows_title'), windows_title = opt_getfiles.windows_title; end


% create java object
filechooser = javax.swing.JFileChooser(pwd);
filechooser.setMultiSelectionEnabled(true);
filechooser.setFileSelectionMode(filechooser.FILES_ONLY);

% manage file filter
switch(file_filter)
    case 'img'
        filechooser.addChoosableFileFilter(imgFilter)
        filechooser.setAcceptAllFileFilterUsed(false)
    case 'image'
        filechooser.addChoosableFileFilter(ImageFilter)
        filechooser.setAcceptAllFileFilterUsed(false)
    case ''
        filechooser.setAcceptAllFileFilterUsed(true)
end      

% set title name for the selection window
filechooser.setDialogTitle(windows_title)

% open selection window
selectionStatus = filechooser.showOpenDialog(com.mathworks.mwswing.MJFrame); 

if selectionStatus == filechooser.APPROVE_OPTION
    pathname = [char(filechooser.getCurrentDirectory.getPath), ...
                java.io.File.separatorChar];
    selectedfiles = filechooser.getSelectedFiles;
    for k = 1:1:size(selectedfiles)
        filenames(k) = selectedfiles(k).getName;
    end
    filenames = char(filenames);  
else
    pathname = pwd;
    filenames = 0;
end

% End of code