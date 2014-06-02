function julien_spatialLP
% Spatial LP filter
%___________________________________________________________________________
%
% 
%
% Uses:
%
% 
% Inputs
%
% *.img conforming to SPM data format (see 'Data')
%
% Outputs
%
% The filtered images are written to the same subdirectories as the 
% original *.img and are prefixed with a 'LP' (i.e. s*.img)
%
%__________________________________________________________________________
% @(#)julien_spatialLP.m	2.11 03/03/04

% get filenames and kernel width
%----------------------------------------------------------------------------
SPMid = spm('FnBanner',mfilename,'2.11');
[Finter,Fgraph,CmdLine] = spm('FnUIsetup','Smooth');
spm_help('!ContextHelp',mfilename);

s     = spm_input('smoothing {FWHM in mm}',1);
P     = spm_get(Inf,'IMAGE','select scans');
n     = size(P,1);

% implement the convolution
%---------------------------------------------------------------------------
spm('Pointer','Watch');
spm('FigName','Smooth: working',Finter,CmdLine);
spm_progress_bar('Init',n,'Smoothing','Volumes Complete');
for i = 1:n
	Q = deblank(P(i,:));
	[pth,nm,xt,vr] = fileparts(deblank(Q));
	U = fullfile(pth,['s' nm xt vr]);
	spm_smooth(Q,U,s);
	spm_progress_bar('Set',i);
end
spm_progress_bar('Clear',i);
spm('FigName','Smooth: done',Finter,CmdLine);
spm('Pointer');
