function j_dmri_autoscale(fname_nifti,fname_dicom,opt)
% =========================================================================
% Re-scale NIFTI files based on DICOM headers.
% 
% 
% INPUT
% fname_nifti			string. No need to put the extension.
% fname_dicom			string. path+file prefix of DCM data.
% (opt)
%   fname_log			string.  log for processing.
% 
% OUTPUT
% (-)
%
% EXAMPLE
% 
%	j_dmri_autoscale('data','/Users/julien/MRI/dicom/connectome/HCP_034/968000-000019-*.dcm')
% 
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2012-03-12
% 2012-03-18: enable inputs
% 2012-03-18: doesn't use MATLAB's dicominfo any more
% =========================================================================
dbstop if error

% Check parameters
if nargin<2, help j_dmri_autoscale, return, end
if ~exist('opt'), opt = []; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log, else fname_log = 'log_j_dmri_autoscale.txt'; end




% delete log file
if exist(fname_log), delete(fname_log), end

j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_dmri_autoscale.m'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])



% open DICOM headers
% =========================================================================

% Get file names
fname_nifti=sct_tool_remove_extension(fname_nifti,1);
j_disp(fname_log,['\nGet file names...'])
list_fname = dir(fname_dicom);
nb_files = size(list_fname,1);
j_disp(fname_log,['.. Number of files: ',num2str(nb_files)])

% get path
j_disp(fname_log,['\nGet path...'])
path_dicom = fileparts(fname_dicom);
j_disp(fname_log,['.. ',path_dicom])

% get scaling info
j_disp(fname_log,['\nRead dicom to get scaling factors...'])
scaling_factor = zeros(1,nb_files);
for i_file = 1:nb_files
	fname = [path_dicom,filesep,list_fname(i_file).name];
	fid = fopen(fname,'r','ieee-le');
	fseek(fid,1,'bof');
	dcm = fread(fid,20000,'*char')';
    fclose(fid);
	ind_scale = strfind(dcm,'Scale Factor');
 	scaling_factor(i_file) = sscanf(dcm(ind_scale+14:ind_scale+20),'%f');
% 	hdr = dicominfo(fname);
% 	scaling_factor(i_file) = str2num(hdr.ImageComments(15:21));
	j_disp(fname_log,['.. ',list_fname(i_file).name,' --> Scaling factor = ',num2str(scaling_factor(i_file))])
end

% save to mat file
save scaling_factor scaling_factor



% Adjust scaling on NIFTI file
% =========================================================================

% Split NIFTI file
j_disp(fname_log,['\nSplit NIFTI file...'])
cmd = ['fslsplit ',fname_nifti,' tmp.data_splitT_ -t'];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
numT = j_numbering_local(nb_files,4,0);

% Adjust scaling for each file
j_disp(fname_log,['\nAdjust scaling for each file...'])
for i_file = 1:nb_files
	fname_nifti_split = ['tmp.data_splitT_',numT{i_file}];
	fname_nifti_split_scaled = ['tmp.data_splitT_scaled_',numT{i_file}];	
	cmd = ['fslmaths ',fname_nifti_split,' -div ',num2str(scaling_factor(i_file)),' ',fname_nifti_split_scaled,' -odt float'];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
end



% Merge back data
j_disp(fname_log,['\nMerge back...'])
fname_nifti_scaled = [fname_nifti,'_scaled'];
cmd = ['fslmerge -t ',fname_nifti_scaled];
for iT = 1:nb_files
	cmd = cat(2,cmd,[' tmp.data_splitT_scaled_',numT{iT}]);
end
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(fname_log,['.. File created: ',fname_nifti_scaled])


% delete temporary files
delete 'tmp.*'


% display time
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['\n'])












% =========================================================================
% =========================================================================
% =========================================================================
function varargout = j_numbering_local(varargin)


% initialization
if (nargin<1) help j_numbering; return; end
max_numbering = varargin{1};
if (nargin<2)
    nb_digits = length(num2str(max_numbering));
else
    nb_digits = varargin{2};
    % check number of digits
    if (nb_digits<length(num2str(max_numbering)))
        error('Number of digits too small!!!');
        return
    end
end
if (nargin<3)
    starting_value = 1;
else
    starting_value = varargin{3};
end
if (nargin<4)
    output_format = 'cell';
else
    output_format = varargin{4};
end

% generate numbering
out_numbering = cell(max_numbering,1);
number = starting_value;
for iNumber=1:max_numbering
    % write number
    number_string = num2str(number);
    % fill with zeros
    for iDigits=1:nb_digits-length(number_string)
        number_string = strcat('0',number_string);
    end
    out_numbering{iNumber} = number_string;
    number = number + 1;
end

if strcmp(output_format,'array')
    out_numbering_tmp = out_numbering;
    clear out_numbering
    for i=1:size(out_numbering_tmp,1)
		out_numbering(i,:) = out_numbering_tmp{i};
    end
	clear out_numbering_tmp
end

% output
varargout{1} = out_numbering;
