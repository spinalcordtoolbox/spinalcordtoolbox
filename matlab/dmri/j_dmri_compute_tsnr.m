function j_dmri_compute_tsnr(fname_data,fname_bvecs,opt)
% =========================================================================
% Compute TSNR of DW data. Uses FSL tools.
% 
% 
% INPUT
% fname_data		string. No need to put the extension.
% fname_bvecs
% (opt)
%   moco			0 | 1*	 moco using FLIRT.
%   detrend			0 | 1*   detrend data.
%   fname_tsnr		string.  file name tsnr
%   fname_log		string.  log for processing.
%
% OUTPUT
% (-)
%
% Example:   j_compute_tsnr('diff','tw_SNR_Ydir_5nav.txt')
% 
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2012-03-13
% =========================================================================



if nargin<2, help j_dmri_compute_tsnr, return, end
if ~exist('opt'), opt = []; end
if isfield(opt,'moco'), moco_do = opt.moco, else moco_do = 1; end
if isfield(opt,'detrend'), detrend_do = opt.detrend, else detrend_do = 1; end
if isfield(opt,'fname_tsnr'), fname_tsnr = opt.fname_tsnr, else fname_tsnr = 'tsnr'; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log, else fname_log = 'log_j_compute_tsnr.txt'; end




% delete log file
if exist(fname_log), delete(fname_log), end

j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_dmri_compute_tsnr.m'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])


% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. Input data:            ',fname_data])
j_disp(fname_log,['.. bvecs file:            ',fname_bvecs])
j_disp(fname_log,['.. moco_do:               ',num2str(moco_do)])
j_disp(fname_log,['.. detrend_do:            ',num2str(detrend_do)])
j_disp(fname_log,['.. fname_tsnr:            ',fname_tsnr])
j_disp(fname_log,['.. fname_log:             ',fname_log])


% find where are the b=0 images
j_disp(fname_log,['\nIdentify b=0 images...'])
bvecs = textread(fname_bvecs);
nb_dirs = size(bvecs,1);
index_b0 = [];
for it = 1:nb_dirs
	if ~sum(bvecs(it,:)~=[0 0 0])
		index_b0 = cat(1,index_b0,it);
	end
end
j_disp(fname_log,['.. Index of b=0 images: ',num2str(index_b0')])
nb_b0 = length(index_b0);


% find where are the DW images
j_disp(fname_log,['\nIdentify DW images...'])
index_dwi = [];
for it = 1:nb_dirs
	if sum(bvecs(it,:)~=[0 0 0])
		index_dwi = cat(1,index_dwi,it);
	end
end
j_disp(fname_log,['.. Index of DW images: ',num2str(index_dwi')])
nb_dwi = length(index_dwi);


% split into T dimension
j_disp(fname_log,['\nSplit along T dimension...'])
cmd = ['fslsplit ',fname_data,' tmp.data_splitT'];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
numT = j_numbering_local(nb_dirs,4,0);


% Merge DWI images
j_disp(fname_log,['\nMerge DWI images...'])
fname_dwi_merge = 'tmp.dwi';
cmd = ['fslmerge -t ',fname_dwi_merge];
for iT = 1:nb_dwi
	cmd = cat(2,cmd,[' tmp.data_splitT',numT{index_dwi(iT)}]);
end
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(fname_log,['.. File created: ',fname_dwi_merge])
% update file name
file_name = fname_dwi_merge;


if moco_do
	% Motion correction
	j_disp(fname_log,['\nMotion correction...'])
	cmd = ['mcflirt -in ',fname_dwi_merge,' -out tmp.dwi_moco -sinc_final -dof 6'];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	j_disp(fname_log,['.. File created: tmp.dwi_moco'])
	% update file name
	file_name = 'tmp.dwi_moco';
end


% Open NIFTI file
j_disp(fname_log,['\nOpen DWI images...'])
[data,dims,scales,bpp,endian] = read_avw(file_name);
[nx ny nz nt] = size(data);


% reshape
data2d = reshape(data,nx*ny*nz,nb_dwi);
clear data

% % Remove b=0 images
% j_disp(fname_log,['\nRemove b=0 images...'])
% data = img(:,:,:,index_dwi);

% compute TSNR
j_disp(fname_log,['\nCompute TSNR...'])
tsnr = zeros(1,nx*ny*nz);
nb_voxels = nx*ny*nz;
i_progress = 0;
pourcentage = 10;
for i_vox = 1:nb_voxels

	data1d = data2d(i_vox,:);

	% detrend data
	if detrend_do
		 data1d = detrend(data1d,'linear') + mean(data1d);
	end 
		
	% compute TSNR
	tsnr(i_vox) = mean(data1d) / std(data1d);
	
	% display progress
	if i_progress > nb_voxels/10;
		j_disp(fname_log,['.. ',num2str(pourcentage),'/100'])
		pourcentage = pourcentage + 10;
		i_progress = 0;
	else
		i_progress = i_progress+1;
	end
end


% Write TSNR
j_disp(fname_log,['\n\nWrite TSNR...'])
tsnr3d = reshape(tsnr,nx,ny,nz);
save_avw(tsnr3d,fname_tsnr,'s',scales);
j_disp(fname_log,['.. File created: ',fname_tsnr])

% Copy geometry information
j_disp(fname_log,['\nCopy geometry information...'])
cmd = ['fslcpgeom ',fname_data,' ',fname_tsnr];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end


% Remove temporary files
j_disp(fname_log,['\nRemove temporary files...'])
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





