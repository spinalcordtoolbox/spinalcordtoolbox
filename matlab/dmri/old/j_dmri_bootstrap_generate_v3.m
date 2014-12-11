% =========================================================================
% FUNCTION
% j_dmri_bootstrap_generate.m
%
% Generate bootstrap data out of a set of multiple averaging data.
% Call this function with batch_bootstrap.m
% 
% Julien Cohen-Adad <>
% 2009-10-30: created
% 2012-02-26: implementation residual bootstrap (Berman et al. Neuroimage)
% =========================================================================
function bootstrap = j_dmri_bootstrap_generate_v3(bootstrap)




% INITIALIZATION
if ~exist('opt'), opt = []; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log, else fname_log = 'log_j_dmri_reorganize_data.txt'; end
if isfield(opt,'outputtype'), outputtype = opt.outputtype, else outputtype = 'NIFTI'; end



% =========================================================================
% START THE SCRIPT
% =========================================================================



% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_dmri_bootstrap_generate_v3'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])



% Find which SHELL is running
j_disp(fname_log,['\nFind which SHELL is running...'])
[status result] = unix('echo $0');
if ~isempty(findstr(result,'bash'))
        dmri.shell = 'bash';
elseif ~isempty(findstr(result,'tsh'))
        dmri.shell = 'tsh';
elseif ~isempty(findstr(result,'tcsh'))
        dmri.shell = 'tcsh';
else    
        j_disp(dmri.log,['.. Failed to identify shell. Using default.'])
        dmri.shell = 'bash';
end     
j_disp(fname_log,['.. Running: ',dmri.shell])



% FSL output
if strcmp(dmri.shell,'bash')
        fsloutput = ['export FSLOUTPUTTYPE=',outputtype,'; ']; % if running BASH
elseif strcmp(dmri.shell,'tsh') || strcmp(dmri.shell,'tcsh')
        fsloutput = ['setenv FSLOUTPUTTYPE ',outputtype,'; ']; % if you're running C-SHELL
else
        error('Check SHELL field.')
end



% load mask
fname_mask = [bootstrap.nifti.path,bootstrap.file_mask];
disp(['-> Mask used to crop the data: "',fname_mask,'"'])
j_progress('Load mask .....................................')
mask = read_avw(fname_mask);
if length(mask)==1, error('CHECK FILE NAME FOR THE MASK! Exit program.'); end
j_progress(1)


% Find the size of the mask
for i=1:size(mask,3)
    [x y] = find(mask(:,:,i));
    if ~isempty(x) & ~isempty(y)
        index_xmin = min(x);
        index_xmax = max(x);
        index_ymin = min(y);
        index_ymax = max(y);
        z(i) = i;
        index_zmin = min(find(z));
        index_zmax = max(find(z));
    end
end
nx_tmp = index_xmax-index_xmin+1;
ny_tmp = index_ymax-index_ymin+1;
nz_tmp = index_zmax-index_zmin+1;

% find orientation of the mask (according to MNI template)
j_progress('Find orientation of the mask ..................')
if nx_tmp==1
	orientation = 'coronal';
	nx = index_ymax-index_ymin+1;
	ny = index_zmax-index_zmin+1;
elseif ny_tmp==1
	orientation = 'sagittal';
elseif nz_tmp==1
	orientation = 'axial';
	nx = index_xmax-index_xmin+1;
	ny = index_ymax-index_ymin+1;
end
j_progress(1)
disp(['-> Orientation is: ',orientation])


% get coordinates of the mask (to crop it later)
coord_roi = [num2str(index_xmin-1),' ',num2str(nx),' ',num2str(index_ymin),' ',num2str(ny),' ',num2str(index_zmin-1),' 1 '];
disp(['-> Cropping coordinates are: ',num2str(coord_roi)])

% Crop data
disp(['-> Number of folders is: ',num2str(bootstrap.nb_folders)])
if bootstrap.crop
	j_progress('Crop original data ............................')
	for i_folder = 1:bootstrap.nb_folders
		fname_data = [bootstrap.nifti.path,bootstrap.nifti.folder{i_folder},bootstrap.nifti.file_data];
		fname_data_crop = [bootstrap.nifti.path,bootstrap.nifti.folder{i_folder},bootstrap.file_data_crop];
% 		[data,dims,scales,bpp,endian] = read_avw(fname_data);
% 		data_crop = data(index_xmin:index_xmax,index_ymin:index_ymax,index_zmin:index_zmax,:);
% 		save_avw(data_crop,fname_data_crop,'d',scales);
		cmd = [fsloutput,'fslroi ',fname_data,' ',fname_data_crop,' ',coord_roi];
		unix(cmd);
		j_progress(i_folder/bootstrap.nb_folders)
	end
end
clear data

% crop the BET mask of the image
j_progress('Crop the mask .................................')
fname_mask_bet = [bootstrap.nifti.path,bootstrap.nifti.folder{1},bootstrap.nifti.file_mask];
fname_mask_bet_crop = [bootstrap.nifti.path,bootstrap.nifti.file_mask_crop];
cmd = [fsloutput,'fslroi ',fname_mask_bet,' ',fname_mask_bet_crop,' ',coord_roi];
unix(cmd);
j_progress(1)

% get the dimensions
j_progress('Get number of volumes in each set .............')
bvals = textread([bootstrap.nifti.path,bootstrap.nifti.folder{1},bootstrap.nifti.file_bvals]);
nt = length(bvals);
j_progress(1)
disp(['-> Number of volumes is: ',num2str(nt)])

% create write folder
j_progress('Create write folder ...........................')
path_write = [bootstrap.path];
if ~exist(path_write), mkdir(path_write), end
j_progress(1)
disp(['-> Output folder is: ',path_write])

% open cropped images
j_progress('Open cropped data .............................')
data4d = zeros(nx,ny,nt,bootstrap.nb_folders);
for i_folder = 1:bootstrap.nb_folders
	fname_data_crop = [bootstrap.nifti.path,bootstrap.nifti.folder{i_folder},bootstrap.file_data_crop];
% 	opt.output = 'nifti';
% 	hdr = squeeze(read_avw(fname_data_crop));
% [data,dims,scales,bpp,endian]
	[data4d(:,:,:,i_folder),dims,scales,bpp,endian] = read_avw(fname_data_crop);
% 	data4d(:,:,:,i_folder) = hdr;
	j_progress(i_folder/bootstrap.nb_folders)
end

% number of bootstraps
nb_bootstraps = bootstrap.nb_bootstraps;
num = j_numbering(nb_bootstraps,4,1);

% select bootstrap method
switch bootstrap.method
	
case 'regular'
	
	j_progress('Bootstrap data using "repetition bootstrap" ...')

	for i_bootstrap = 1:nb_bootstraps
		% create volume
		data3d = zeros(nx,ny,nt);
		% loop over each dimension
		for ix = 1:nx
			for iy = 1:ny
				for it = 1:nt
					% randomly pick one value among the original data
					i_data = ceil(rand*bootstrap.nb_folders);
					data3d(ix,iy,it) = data4d(ix,iy,it,i_data);
				end
			end
		end
		% write new image
		file_write = ['data_',num{i_bootstrap}];
		fname_write{i_bootstrap} = [path_write,file_write];
	% 	hdr_new = hdr;
	% 	hdr_new.img = data3d;
	% 	j_save_untouch_nii(hdr_new,fname_write{i_bootstrap});
		save_avw(data3d,fname_write{i_bootstrap},'d',scales);
		j_progress(i_bootstrap/nb_bootstraps)
	end

case 'bootknife'
	
	j_progress('Bootstrap data using "repetition bootknife" ...')
	% For the repetition bootknife, one measurement is randomly discarded 
	% prior to generating new bootstrap datasets, yielding a total number
	% of (n-1) repetitions. Then, one measurement is randomly replicated
	% yielding a total number of n measurements. From these n measurements,
	% bootstrap datasets are generated N times (e.g. N=500) exactly the
	% same way it is done using repetition bootstrap. This procedure is
	% done for each diffusion direction and each voxel independently.
	%
	% first, generate a new data4d variable, using jacknife procedure
	for ix = 1:nx
		for iy = 1:ny
			for it = 1:nt
				% get indices for all repetitions
				index = [1:bootstrap.nb_folders];
				% randomly remove one index
				index(ceil(rand*bootstrap.nb_folders)) = 0;
				index_reduced = nonzeros(index);
				% randomly select one index from the n-1 distribution
				random_ind = ceil(rand*(bootstrap.nb_folders-1));
				% add it to the reduced distribution
				index_new = cat(1,index_reduced,index_reduced(random_ind));
				% modify the data4d variable
				data4d_new(ix,iy,it,:) = data4d(ix,iy,it,index_new);
			end
		end
	end

	% Bootstrap data
	for i_bootstrap = 1:nb_bootstraps
		% create volume
		data3d = zeros(nx,ny,nt);
		% loop over each dimension
		for ix = 1:nx
			for iy = 1:ny
				for it = 1:nt
					% randomly pick one value among the original data
					i_data = ceil(rand*bootstrap.nb_folders);
					data3d(ix,iy,it) = data4d(ix,iy,it,i_data);
				end
			end
		end
		% write new image
		file_write = ['data_',num{i_bootstrap}];
		fname_write{i_bootstrap} = [path_write,file_write];
	% 	hdr_new = hdr;
	% 	hdr_new.img = data3d;
	% 	j_save_untouch_nii(hdr_new,fname_write{i_bootstrap});
		save_avw(data3d,fname_write{i_bootstrap},'d',scales);
		j_progress(i_bootstrap/nb_bootstraps)
	end

case 'residual'
	
	% Reshape
	data2d = reshape(data4d,nx*ny,nt);
	clear data4d
	nb_voxels = size(data2d,1);
	
	% create data Bootstrap
	data2d_bootstrap = zeros(nb_bootstraps,nx*ny,nt-1);
	
	% remove b=0 at the beginning
	data2d = data2d(:,2:end);
	
	% open bvecs
	fname_bvecs = [bootstrap.nifti.path,bootstrap.nifti.folder{1},bootstrap.nifti.file_bvecs];
	bvecs = textread(fname_bvecs);

	% Define samplings for inputOrientationSet for order
% 	j_disp(opt.fname_log,['\nDefine sampling for SH order...'])
	scheme = j_gen_scheme(bvecs, bootstrap.qball.order_odf);

	% Estimate Q-Ball using spherical harmonics
	for i_voxel=1:nb_voxels
		S = data2d(i_voxel,:);
		S_SH = j_amp2SH(S',scheme);
		S_fit = j_SH2amp(S_SH,scheme)';
		
		% compute residuals
		r = S-S_fit;
		
		% bootstrap residuals
		for i_bootstrap = 1:nb_bootstraps

			% permute residuals
			ind_perm = randperm(nt-1);
			r_bootstrap = r(ind_perm);

			% generate sythetic bootstrap data
			data2d_bootstrap(i_bootstrap,i_voxel,:) = S_fit + r_bootstrap;
		end
	end
	
	% write bootstrap images
	for i_bootstrap = 1:nb_bootstraps

		% reshape data
		data3d = reshape(data2d_bootstrap(i_bootstrap,:,:),nx,ny,nt-1);
		
		% write data
		file_write = ['data_',num{i_bootstrap}];
		fname_write{i_bootstrap} = [path_write,file_write];
		save_avw(data3d,fname_write{i_bootstrap},'d',scales);
	end	
end


% save structure
bootstrap.bootstrap.nb_folders = bootstrap.nb_folders;
bootstrap.nx = nx;
bootstrap.ny = ny;
bootstrap.nt = nt;
bootstrap.orientation = orientation;
bootstrap.fname_data = fname_write;
% bootstrap.fname_qball = fname_qball;
bootstrap.num = num;
% save([bootstrap.nifti.path,bootstrap.struct_process],'dmri');
save([bootstrap.path,bootstrap.file_struct,bootstrap.file_struct_suffixe],'bootstrap');

