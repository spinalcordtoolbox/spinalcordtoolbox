function j_mri_average(fname_data, opt)
% =========================================================================
% 
% Average several 4D datasets together. Write new file locally.
% 
% 
% INPUT
% -------------------------------------------------------------------------
% fname_data			cell. Each cell contains data file name. DO NOT PUT THE FILE EXTENSION!!
% (opt) 
%   fname_average		string.		Default='data_average'
%   split_data			0* | 1		If too memory intensive, split data
%   outputtype			NIFTI | NIFTI_GZ
%	fname_log			string
% -------------------------------------------------------------------------
% 
% OUTPUT
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------
% 
%   Example
%   j_dmri_average
%
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2012-01-09: Created
%
% =========================================================================

% PARAMETERS


% Check number of arguments
if nargin < 1
	disp('Not enought arguments. Type: help j_mri_average')
	return
end

% INITIALIZATION
dbstop if error; % debug if error
if ~exist('opt'), opt = []; end
if isfield(opt,'fname_average'), fname_average = opt.fname_average, else fname_average = 'data_average'; end
if isfield(opt,'split_data'), split_data = opt.split_data, else split_data = 1; end
if isfield(opt,'outputtype'), outputtype = opt.outputtype, else outputtype = 'NIFTI'; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log, else fname_log = 'log_j_mri_average.txt'; end


% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_mri_average'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])


% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. fname_average:     ',fname_average])
j_disp(fname_log,['.. split_data:        ',num2str(split_data)])
j_disp(fname_log,['.. Output type:       ',outputtype])
j_disp(fname_log,['.. log file:          ',fname_log])



% Find which SHELL is running
j_disp(fname_log,['\nFind which SHELL is running...'])
[status result] = unix('echo $0');
if ~isempty(findstr(result,'bash'))
	shell = 'bash';
elseif ~isempty(findstr(result,'tsh'))
	shell = 'tsh';
elseif ~isempty(findstr(result,'tcsh'))
	shell = 'tcsh';
else
	j_disp(fname_log,['.. Failed to identify shell. Using default.'])
	shell = 'bash';
end
j_disp(fname_log,['.. Running: ',shell])



% FSL output
if strcmp(shell,'bash')
	fsloutput = ['export FSLOUTPUTTYPE=',outputtype,'; ']; % if running BASH
	fsloutput_temp = ['export FSLOUTPUTTYPE=NIFTI; '];
elseif strcmp(shell,'tsh') || strcmp(shell,'tcsh') 
	fsloutput = ['setenv FSLOUTPUTTYPE ',outputtype,'; ']; % if you're running T-SHELL
	fsloutput_temp = ['setenv FSLOUTPUTTYPE NIFTI; '];
end



% get number of files
j_disp(fname_log,['\nGet number of files...'])
nb_files = length(fname_data);
j_disp(fname_log,['.. ',num2str(nb_files)])



% get data dimensions
j_disp(fname_log,['\nGet dimensions of the data...'])
cmd = ['fslsize ',fname_data{1}]; [status result] = unix(cmd); if status, error(result); end
dims = j_mri_getDimensions(result);
nx = dims(1);
ny = dims(2);
nz = dims(3);
nt = dims(4);
j_disp(fname_log,['.. ',num2str(nx),' x ',num2str(ny),' x ',num2str(nz),' x ',num2str(nt)])
clear img



% If too memory intensive, split data
if split_data

	% Split data
	folder_dataSplit = {};
	for i_file = 1:nb_files
		j_disp(fname_log,['\nSplit file ',num2str(i_file),'/',num2str(nb_files),'...'])
		folder_dataSplit{i_file} = ['tmp.split',num2str(i_file),filesep];
		if ~exist(folder_dataSplit{i_file}), mkdir(folder_dataSplit{i_file}), end
		cmd = [fsloutput_temp,'fslsplit ',fname_data{i_file},' ',folder_dataSplit{i_file},'tmp.data_T'];
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	end
	numT = j_numbering(nt,4,0);

	% Average data
	j_disp(fname_log,['\nAverage data...'])
	folder_merge = ['tmp.splitMerge',filesep];
	if ~exist(folder_merge), mkdir(folder_merge), end
	folder_average = ['tmp.splitAverage',filesep];
	if ~exist(folder_average), mkdir(folder_average), end
	fname_average_split = {};
	for iT=1:nt	
		% merge data
		fname_merge = [folder_merge,'tmp.data_merge_T',numT{iT}];
		cmd = [fsloutput_temp,'fslmerge -t ',fname_merge];
		for i_file = 1:nb_files
			cmd = [cmd,' ',folder_dataSplit{i_file},'tmp.data_T',numT{iT}];
		end
 		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		
		% average data
		fname_average_split{iT} = [folder_average,'tmp.data_average_T',numT{iT}];
		cmd = [fsloutput_temp,'fslmaths ',fname_merge,' -Tmean ',fname_average_split{iT}];
 		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	end	
	
	% Merge data back
	j_disp(fname_log,['\nMerge data back...'])
	cmd = [fsloutput,'fslmerge -t ',fname_average];
	for iT=1:nt
		cmd = [cmd,' ',fname_average_split{iT}];
	end
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	
	% Delete temp files
	j_disp(fname_log,['\nDelete temporary files...'])
	cmd = 'rm -rf tmp.split*';
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	
else
	
	% Allocate memory
	data = zeros(nx,ny,nz,nt,nb_files);
    data_mean = zeros(nx,ny,nz,nt);
    
	% Open data
	for i_file = 1:nb_files

		j_disp(fname_log,['\nOpen file ',num2str(i_file),'/',num2str(nb_files),'...'])
		[data(:,:,:,:,i_file),dims,scales,bpp,endian] = read_avw(fname_data{i_file});

    end
    
    % average data
 	j_disp(fname_log,['\nAverage data...'])
    data_mean = mean(data,5);
    
	% save volume
	j_disp(fname_log,['.. output name: ',fname_average])
	save_avw(data_mean,fname_average,'f',scales(1:3));
	j_disp(fname_log,['\nCopy geometric information from ',fname_data{1},'...'])
	cmd = [fsloutput,'fslcpgeom ',fname_data{1},' ',fname_average,' -d'];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

end

% END FUNCTION
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])
