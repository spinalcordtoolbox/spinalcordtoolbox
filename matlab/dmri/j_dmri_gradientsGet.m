% =========================================================================
% FUNCTION
% j_dmri_gradientsGet.m
%
% Get diffusion gradient list.
%
% INPUT
% (opt)					structure
%	gradient_format		'fsl'*, 'medinria'
%	path_read			string. Could be:
%						1) Full path where DICOMs are located. E.g.: '/Users/julien/mri/data/subj01/'
%						2) Path + part of the file name. E.g.: '/Users/julien/mri/data/subj01/611000-000048-0*.dcm'
%	path_write
%	correct_bmatrix
%	file_bvecs
%	file_bvals
%   verbose
%
% OUTPUT
% gradient				structure
%	bvecs				nx3
%	bvals				nx1
%
% julien cohen-adad <jcohen@nmr.mgh.harvard.edu>
% 2009-08-31: Created
% 2011-10-09: Memory issue fixed when using the Matlab flag.
% 
% =========================================================================
function gradient = j_dmri_gradientsGet(opt)


% default parameters
gradient_format			= 'fsl'; % fsl, medinria
path_read				= '.'; % if blank, then manual selection
path_write				= '..'; % if blank, then manual selection (it'll generate folder for each run)
correct_bmatrix			= 0;
file_bvecs				= 'bvecs';
file_bvals				= 'bvals';
method_default					= 'matlab'; % 'matlab'* (using dicomdir), 'julien' (using custom parser)
verbose					= 1;

% user parameters
if ~exist('opt'), opt = []; end
if isfield(opt,'gradient_format'), gradient_format = opt.gradient_format; end
if isfield(opt,'path_read'), path_read = opt.path_read; end
if isfield(opt,'path_write'), path_write = opt.path_write; end
if isfield(opt,'correct_bmatrix'), correct_bmatrix = opt.correct_bmatrix; end
if isfield(opt,'file_bvecs'), file_bvecs = opt.file_bvecs; end
if isfield(opt,'file_bvals'), file_bvals = opt.file_bvals; end
if isfield(opt,'verbose'), verbose = opt.verbose; end
if isfield(opt,'method'), method = opt.method; else, method = method_default; end


% 
% 
% folder_write                = ''; % if empty, put the run name + protocol (e.g. 08-ep2d_bold_TE20)
% suffixe_write               = '';
% 
% % misc
% nb_averages                 = 0; % if 0, find automatically
% output_format				= 'nii'; % img, nii
% nb_files                    = 0; % if 0, find automatically
% nb_slices                   = 0; % nb slices per volumes. If 0, find automatically
% format_spm                  = 0; % generate a .mat file
% max_distance_vector			= 0.001; % euclidean distance between two gradient vectors considered as being the same
% 
% % motion correction
% moco						= 'O'; % O,N
% type_moco					= 'D6'; %M pour Medic/ D2 ou D3 ou D6 pour Diff 2dof, 3dof, 6dof
% 
% % misc DTI ONLY
% gradient_file               = 1; % generate a gradient file for MedINRIA
% rot_angle					= [0 180 0]; % rotate angle according to [x y z] matrix (given in degree)
% % fname_gradientRead          = 'D:\mes_documents\IRM\DTI\MedINRIA_xxdirections_axial.txt'; % template of gradient file for automatic generation
% % start_gradientReadDir       = 35;
% % stop_gradientReadDir        = 36;
% % prefixe_dicom_b0            = 'ep_b0'; % works for Siemens VA25
% max_nb_b0                   = 40;
% max_nb_directions           = 99;
% 
% % don't touch
% nb_directions               = 0; % specify if problem in image sequencing (e.g., when performing physiological triggering using the MGH pulse sequence). Otherwise, put 0.
% nb_b0                       = 0;
% prefixe_write               = '';
% path_meanB0_relative        = '';
% path_meanB0                 = '';
% path_meanDiff               = '';
% ismosaic                    = 0;
% check_numbering             = 0;
% find_gradients              = 1; % find gradient vectors using DICOM header
% 

% start function
if verbose
	fprintf('\n')
	fprintf('+++ j_dmri_gradientsGet +++')
	fprintf('\n\n')
end

% retrieve source files
if verbose, j_progress('Retrieve files ...........................'); end
switch exist(path_read)
	
	% directory
	case 7
		file_read_char = dir(cat(2,path_read,filesep,'*.dcm'));
	
	% file prefixe
	case 0
		file_read_char = dir(cat(2,path_read));
		path_read = fileparts(path_read);
end
if verbose, j_progress(1); end

% check if there is no dicom extension
if isempty(file_read_char)
	fprintf('-> warning: no DICOM extension\n');
	file_read_char_tmp = dir(cat(2,path_read,filesep,'*'));
	% remove non-DICOM files
	j=1;
	for i=1:size(file_read_char_tmp,1)
		if ~file_read_char_tmp(i).isdir
			file_read_char(j,:) = file_read_char_tmp(i);
			j=j+1;
		end
	end
	clear file_read_char_tmp
end
% convert to cell
for i=1:size(file_read_char,1)
	fname_read{i} = strcat(path_read,filesep,file_read_char(i).name);
end
clear file_read_char path_read
% % retrieve files only (remove directories for instance)
% j = 1;
% for i=1:size(fname_read,2)
% 	if exist(fname_read{i})==2
% 		fname_read_tmp{j} = fname_read{i};
% 		j = j + 1;
% 	end
% end
% clear fname_read
% fname_read = fname_read_tmp;
% clear fname_read_tmp

% retrieve nb_files
nb_files = size(fname_read,2);
if verbose, fprintf('-> %i files\n',nb_files); end

hdr_julien = {};
		
switch method
	
case 'matlab'

	if verbose, j_progress('Read DICOM headers .......................'); end
	for i_file=1:nb_files
		
		% read dicom header
		hdr = dicominfo(fname_read{i_file});

		% get the right ordering for volumes
		acquisition_order(i_file) = hdr.AcquisitionNumber;
		
		% if B0, there is no DICOM field 
		if ~isfield(hdr,'Private_0019_100e')
			hdr_julien{i_file}.PrivateJulien.Diffusion.BValue = 0;
			hdr_julien{i_file}.PrivateJulien.Diffusion.DiffusionGradientDirection  = [0 0 0];
		else
			hdr_julien{i_file}.PrivateJulien.Diffusion.BValue = hdr.Private_0019_100c;
			hdr_julien{i_file}.PrivateJulien.Diffusion.DiffusionGradientDirection  = hdr.Private_0019_100e';
		end
		
		if verbose, j_progress(i_file/nb_files); end
	end
% 	if verbose, j_progress(1); end

case 'julien'

	% read dicom header of all files
	if verbose, j_progress('Read DICOM headers .......................'); end
	for i_file=1:nb_files
		hdr_julien{i_file} = j_dicom_read(fname_read{i_file});
		j_progress(i_file/nb_files)
		
		% get the right ordering for volumes
		acquisition_order(i_file) = hdr_julien{i_file}.AcquisitionNumber;
		
	end
	if verbose, j_progress(1); end
end

% create an array
for i_file = 1:nb_files
	gradient_list(acquisition_order(i_file),:) = hdr_julien{i_file}.PrivateJulien.Diffusion.DiffusionGradientDirection;
	bvalue(acquisition_order(i_file)) = hdr_julien{i_file}.PrivateJulien.Diffusion.BValue;
end

% apply a rigid transformation to account for slice orientation - because
% gradient coordinates are in scanner and not patient space.
if correct_bmatrix
	if verbose, j_progress('Correct b-matrix for slice orientation ...'); end
	gradient_list_t = j_dmri_gradientsTransform(gradient_list,hdr);
	if verbose, j_progress(1); end
else
	gradient_list_t = gradient_list;
end
		
% Generate gradient file
if verbose, j_progress('Generate gradient file ...................'); end
if ~exist(path_write), mkdir(path_write), end
fname_bvecs = [path_write,file_bvecs];
j_dmri_gradientsWrite(gradient_list_t,fname_bvecs,gradient_format);
j_progress(1);
if verbose, fprintf('-> file created: %s\n',fname_bvecs); end

% Generate bvalue file
if verbose, j_progress('Generate bvalue file .....................'); end
fname_bvals = [path_write,file_bvals];
j_dmri_gradientsWriteBvalue(bvalue,fname_bvals,gradient_format);
j_progress(1);
if verbose, fprintf('-> file created: %s\n',fname_bvals); end

% output
gradient.bvecs = gradient_list_t;
gradient.bvals = bvalue';
