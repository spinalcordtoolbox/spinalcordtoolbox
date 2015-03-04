function [sct, status] = sct_dmri_prepare_bvecs(sct)
% =========================================================================
% 
% Prepare bvecs and bvals files for dmri.
% 
% 
% 
% INPUT
% -------------------------------------------------------------------------
% sct
%   (fname_log)				string
% 
% -------------------------------------------------------------------------
%
% OUTPUT
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------
% 
%   Example
%   sct_dmri_prepare_bvecs
%
%
% Julien Cohen-Adad <jcohen@polymtl.ca>
% 2013-06-22: Created
% 2013-09-28: removed DICOM dependence
%
% =========================================================================

% PARAMETERS


% Check number of arguments
if nargin < 1
	disp('Not enought arguments. Type: help sct_dmri_prepare_bvecs')
	return
end

% INITIALIZATION
dbstop if error; % debug if error
status = 0;
if ~exist('sct'), opt = []; end
if isfield(sct,'fname_log'), fname_log = sct.fname_log, else fname_log = 'log_sct_dmri_prepare_bvecs.txt'; end


% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: sct_dmri_prepare_bvecs'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])

% Check parameters
j_disp(fname_log,['\nCheck parameters:'])

% TODO



% % Process bvals bvecs
% if max(strcmp('dmri',data_type))~=0


% % IF USER HAS DICOM INPUTS:
% 
% if ~exist([sct.output_path,'dmri/bvecs.txt']) && strcmp(sct.input_files_type,'dicom');
% 	
% 	% Get gradient vectors
% 	j_disp(fname_log,['\nGet gradient vectors...']);
% 	opt.path_read  = [sct.input_path,sct.dicom.dmri.folder,filesep];
% 	opt.path_write = [sct.output_path,'dmri/'];
% 	opt.file_bvecs = 'bvecs.txt';
% 	opt.file_bvals = 'bvals.txt';
% 	opt.verbose = 1;
% 	if strcmp(sct.dmri.gradients.referential,'XYZ')
% 
% 		opt.correct_bmatrix = 1;
% 
% 	elseif strcmp(sct.dmri.gradients.referential,'PRS')
% 
% 		opt.correct_bmatrix = 0;
% 
% 	end
% 
% 	gradients = j_dmri_gradientsGet(opt);
% 
% else
% 	
% 	% IF  USER HAS NIFTI INPUT
% 	% TODO: test if the file is there.
% end





% OK, NOW WE CAN PREPARE THE BVECS/BVALS FILES.

%---------------------------
% read in gradient vectors
%---------------------------
j_disp(fname_log,['\nRead in gradient vectors...'])

% bvecs

fname_bvecs = [sct.input_path,sct.dmri.folder,sct.dmri.file_bvecs];
sct.dmri.data_bvecs = load(fname_bvecs);
j_disp(fname_log,['.. File bvecs: ',fname_bvecs])


% bvals
fname_bvals = [sct.input_path,sct.dmri.folder,sct.dmri.file_bvals];

if exist(fname_bvals)
	j_disp(fname_log,['.. File bvals: ',fname_bvals])
	sct.dmri.data_bvals = load(fname_bvals);
else
	j_disp(fname_log,['.. !! bvals file is empty. Must be DSI data.'])
end

%---------------------------
% check directions
%---------------------------

j_disp(fname_log,['.. Number of directions: ',num2str(size(sct.dmri.data_bvecs,1))])
if exist(fname_bvals)
	j_disp(fname_log,['.. Maximum b-value: ',num2str(max(sct.dmri.data_bvals)),' s/mm2'])
end

%---------------------------
% flip gradient
%---------------------------
flip = sct.dmri.gradients.flip;
if flip(1)~=1 || flip(2)~=2 || flip(3)~=3
	j_disp(fname_log,['\nFlip gradients...'])
	j_disp(fname_log,['.. flip options: ',num2str(flip)])
	fname_bvecs = [sct.output_path,'dmri/bvecs.txt'];
	gradient = load(fname_bvecs);
	sct.dmri.file_bvecs = [sct.dmri.file_bvecs,'_flip',num2str(flip(1)),num2str(flip(2)),num2str(flip(3))];
	fname_bvecs_new = [sct.output_path,'dmri/',sct.nifti.dmri.file_bvecs];
	fid = fopen(fname_bvecs_new,'w');
	for i=1:size(gradient,1)
		G = [sign(flip(1))*gradient(i,abs(flip(1))),sign(flip(2))*gradient(i,abs(flip(2))),sign(flip(3))*gradient(i,abs(flip(3)))];
		fprintf(fid,'%1.10f %1.10f %1.10f\n',G(1),G(2),G(3));
	end
	fclose(fid);
	j_disp(fname_log,['.. File written: ',fname_bvecs_new])

end

    




% END FUNCTION
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])
