% =========================================================================
% FUNCTION
% j_mri_process_data_v5.m
%
% run batch_dcm2nii.m first. It will generate the 'mri' structure. Supposed
% to work for fmri and mri.
% 
% COMMENTS
% Julien Cohen-Adad 2010-08-21
% =========================================================================
function mri = j_mri_process_data_v5(mri)


% parameters


% =========================================================================
% START THE SCRIPT
% =========================================================================

% initializations
fsloutput = mri.fsloutput;
nb_folders = length(mri.nifti.folder);
mri.nifti.file_data = mri.nifti.file_data_raw;
ext = '.nii';

j_cprintf('red','\nPROCESS MRI DATA (v5)\n')

j_cprintf('blue','\nREORIENT DATA\n')

% reorient data
if mri.reorient.do
	
	disp('Re-orient data according to MNI template:')
	for i_folder = 1:nb_folders
		% build file names
		fname_data = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data];
		% re-orient data
		cmd = [fsloutput,'fslswapdim ',fname_data,' ',mri.nifti.reorient,' ',fname_data];
		j_cprintf('Magenta',[cmd,'\n'])
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
	end
else
	fprintf(1,'Skip this step.\n');
end

% get data dimensions
disp('Get dimensions of the data:')
for i_folder = 1:nb_folders
	fname_data = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data];
	cmd = [fsloutput,'fslsize ',fname_data];
	j_cprintf('Magenta',[cmd,'\n'])
	[status result] = unix(cmd); % run UNIX command
	if status, error(result); end % check error
	dims = j_mri_getDimensions(result);
	mri.nx{i_folder} = dims(1);
	mri.ny{i_folder} = dims(2);
	mri.nz{i_folder} = dims(3);
	mri.nt{i_folder} = dims(4);
	disp(['Folder "',mri.nifti.folder{i_folder},'": ',num2str(dims(1)),'x',num2str(dims(2)),'x',num2str(dims(3)),'x',num2str(dims(4)),' (',num2str(i_folder),'/',num2str(nb_folders),')'])
end










% ====================================================================
% CROP DATA 
% ====================================================================
j_cprintf('black','');
j_cprintf('blue','\nCROP DATA\n');

disp(['-> Cropping method: ',mri.crop.method])
switch (mri.crop.method)

	case 'manual'

	% loop across folders
	for i_folder=1:nb_folders
		% display stuff
 		j_cprintf('black','\n')
		j_cprintf('-black','FOLDER %i/%i: %s',i_folder,nb_folders,mri.nifti.folder{i_folder})
		j_cprintf('black','\n')
		% split the data into Z dimension
		j_progress('Split the data into Z dimension ...............')
		fname_data = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data];
		fname_data_splitZ = [mri.nifti.path,mri.nifti.folder{i_folder},'tmp.data_splitZ'];
		cmd = [fsloutput,'fslsplit ',fname_data,' ',fname_data_splitZ,' -z'];
		disp(['>> ',cmd])
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)
		% split the mask into Z dimension
		j_progress('Split the cropping mask into Z dimension ......')
		fname_mask = [mri.nifti.path,mri.nifti.folder{1},mri.crop.file_crop];
		fname_mask_splitZ = [mri.nifti.path,mri.nifti.folder{i_folder},'tmp.mask_splitZ'];
		cmd = [fsloutput,'fslsplit ',fname_mask,' ',fname_mask_splitZ,' -z'];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)
		% Crop each slice individually
		j_progress('Crop each slice individually ..................')
		numZ = j_numbering(mri.nz{i_folder},4,0);
		for iZ = 1:mri.nz{i_folder}
			% load mask
			fname_mask = [mri.nifti.path,mri.nifti.folder{i_folder},'tmp.mask_splitZ',numZ{iZ}];
			[mask,dims,scales,bpp,endian] = read_avw(fname_mask);
			if length(mask)==1, error('CHECK FILE NAME FOR THE MASK! Exit program.'); end
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
			% in case use wants a bigger/smaller mask
			if mri.crop.manualBoxSize
				index_xmin = index_xmin - round((mri.crop.manualBoxSize-nx_tmp)/2);
				index_ymin = index_ymin - round((mri.crop.manualBoxSize-ny_tmp)/2);
				nx_tmp = mri.crop.manualBoxSize;
				ny_tmp = mri.crop.manualBoxSize;
			end
			% Crop data
			fname_data_splitZ = [mri.nifti.path,mri.nifti.folder{i_folder},'tmp.data_splitZ',numZ{iZ}];
			fname_data_crop_splitZ = [mri.nifti.path,mri.nifti.folder{i_folder},'tmp.data_splitZ_crop',numZ{iZ}];
			cmd = [fsloutput,'fslroi ',fname_data_splitZ,' ',fname_data_crop_splitZ,' ',...
				num2str(index_xmin),' ',...
				num2str(nx_tmp),' ',...
				num2str(index_ymin),' ',...
				num2str(ny_tmp),' ',...
				'0 1'];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error	
		end %  iZ
		j_progress(1)
		disp(['-> Size of the mask: ',num2str(nx_tmp),'x',num2str(ny_tmp)])
		% Merge data along Z
		j_progress('Merge moco b0 along Z dimension ...............')
		fname_data_crop_splitZ = [mri.nifti.path,mri.nifti.folder{i_folder},'tmp.data_splitZ_crop*.*'];
		fname_data_crop = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data,mri.nifti.file_data_crop];
		cmd = [fsloutput,'fslmerge -z ',fname_data_crop,' ',fname_data_crop_splitZ];
		[status result] = unix(cmd);
		if status, error(result); end
		j_progress(1)
		% delete temp files
		delete([mri.nifti.path,mri.nifti.folder{i_folder},'tmp.*'])
	end % i_folder
	% change the default data file name
	mri.nifti.file_data = [mri.nifti.file_data,mri.nifti.file_data_crop];
	disp(['-> File created: ',mri.nifti.file_data])

	case 'box'
		
	disp(['-> Crop size: ',mri.crop.size])
	j_progress('Crop data .....................................')
	for i_folder=1:nb_folders
		fname_data = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data];
		fname_datacrop = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data,mri.nifti.file_data_crop];
		cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',mri.crop.size];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(i_folder/nb_folders)
	end
	% change the default data file name
	mri.nifti.file_data = [mri.nifti.file_data,mri.nifti.file_data_crop];
	disp(['-> File created: ',mri.nifti.file_data])
	
		
	case 'none'
		
	fprintf(1,'Skip this step.\n');
	
end









% Merge data into 4D file to let the program think that this is actually a
% 4D sequence, so that it can use the "intra-run-moco" solution.
if nb_folders>1
	
	j_cprintf('Red',['\n!!! Found more than one folder -> From now on, only use the "averaged" folder.\n'])
	mri.average.do = 1;

	disp('merge all volumes together for motion correction')
	% Build folder name
	j_progress('Build folder name .............................')
	for iFolder=1:nb_folders
		mri.nifti.folder_average	= strcat(mri.nifti.folder_average,mri.nifti.folder{iFolder}(1:end-1),'-'); % average multiple folders
	end
	mri.nifti.folder_average = strcat(mri.nifti.folder_average(1:end-1),'/');
	j_progress(1)
	
	% create folder (if does not already exist)
	if ~exist(strcat(mri.nifti.path,mri.nifti.folder_average))
		j_progress('Create folder .................................')
		mkdir(strcat(mri.nifti.path,mri.nifti.folder_average));
		j_progress(1)
	end
	disp(['Folder created: "',mri.nifti.folder_average,'"'])

	j_progress('Merge data into 4D file .......................')
	fname_data_3d = [];
	for iFolder = 1:nb_folders
		fname_data_3d = strcat(fname_data_3d,[' ',mri.nifti.path,mri.nifti.folder{iFolder},mri.nifti.file_data]);
	end
	fname_data_merged = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data,'_merged'];
	cmd = [fsloutput,'fslmerge -t ',fname_data_merged,' ',fname_data_3d];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(1)

	j_cprintf('Green',['Change variable: mri.nt{1} = ',num2str(nb_folders),'\n'])
	mri.nt{1} = nb_folders;
	j_cprintf('Green',['Change variable: nb_folders = 1\n'])
	nb_folders = 1;
	j_cprintf('Green',['Change variable: mri.nifti.folder{1} = mri.nifti.folder_average\n'])
	mri.nifti.folder{1} = mri.nifti.folder_average;
% 	
% 	mri.nifti.file_data_old = mri.nifti.file_data;
	mri.nifti.file_data = [mri.nifti.file_data,'_merged'];
end
	








% ====================================================================
% INTER-RUN MOTION CORRECTION 
% ====================================================================
% Perform inter-run motion correction, by registering the first b=0 found
% in runs #2 -> #last to the first b=0 found in run #1.
% For each folder, write a mat file that will be used during the intra-run
% motion correction (as an initialization matrix).
j_cprintf('blue','\nINTER-RUN MOTION CORRECTION\n')

if mri.moco_inter.do

	for iFolder = 2:nb_folders
		disp(['Folder #',num2str(iFolder)])
		fname_target = [mri.nifti.path,mri.nifti.folder{1},mri.nifti.file_data];
		fname_source = [mri.nifti.path,mri.nifti.folder{iFolder},mri.nifti.file_data];
		fname_source_reg = [mri.nifti.path,mri.nifti.folder{iFolder},mri.nifti.file_data,'_reg'];
		cmd = [fsloutput,'flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_source_reg,' -cost ',mri.moco_inter.cost,' -dof 9 -interp sinc'];
		j_cprintf('Magenta',[cmd,'\n'])
		j_progress('Co-register to the first folder ...............')
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)
	end

	% change file name of the first folder
	j_progress('Change file name of the first folder ..........')
	fname_source = [mri.nifti.path,mri.nifti.folder{1},mri.nifti.file_data,ext];
	fname_source_reg = [mri.nifti.path,mri.nifti.folder{1},mri.nifti.file_data,'_reg',ext];
	copyfile(fname_source, fname_source_reg);
	j_progress(1)
	
	% Change Nifti file name
	mri.nifti.file_data = [mri.nifti.file_data,'_reg'];
	disp(['-> File(s) created: ',mri.nifti.file_data,ext])

else
	
	fprintf(1,'Skip this step.\n');

end










% ====================================================================
% INTER-SUBJECT MOTION CORRECTION 
% ====================================================================
% Perform inter-session motion correction, by registering the first b=0 of
% run #1 into the b=0 image specified by the user.
% For each folder, write a mat file that will be used during the intra-run
% motion correction (as an initialization matrix).
j_cprintf('blue','\nINTER-SUBJECT MOTION CORRECTION\n')

if mri.moco_session.do

% 	% display stuff
% 	disp(['-> Target series is: "',mri.moco_session.fname,'"'])
% 
% 	% Estimate subject motion between sessions
% 	j_progress('Estimate inter-session motion .................')
% 	fname_target = [mri.moco_session.fname];
% 	fname_source = [mri.nifti.path,mri.nifti.folder{1},mri.moco_inter.file_b0];
% 	fname_moco_session = [dmri.nifti.path,dmri.nifti.folder{1},dmri.moco_session.file_moco];
% 	fname_mat_session = [dmri.nifti.path,dmri.nifti.folder{1},dmri.moco_session.file_mat];
% 	cmd = ['flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_moco_session,' -omat ',fname_mat_session,' -cost ',dmri.moco_session.cost,' -dof 6'];
% 	[status result] = unix(cmd);
% 	j_progress(1)
	
else
	
	fprintf(1,'Skip this step.\n');

end










% ====================================================================
% INTRA-RUN MOTION CORRECTION 
% ====================================================================
j_cprintf('black','');
j_cprintf('blue','\nINTRA-RUN MOTION CORRECTION\n');

% motion correction is done on the "averaged" folder. If exist.
% if nb_folders>1
% 	disp('-> Found more than one folder: Motion correction will be performed on the "averaged" folder.')
% 	folder = mri.nifti.folder_average;
% else
% 	disp('-> Only found one folder.')
	folder = mri.nifti.folder{1};
% end

% disp(['-> Motion correction method: "',mri.moco_intra.method,'"'])
% disp(['-> Crop data: ',num2str(mri.moco_intra.crop)])

if ~strcmp(mri.moco_intra.method,'none')
	% if no ref file is provided
	if isempty(mri.moco_intra.ref)
		disp('Compute target image:')
		% Get the first #n time series
		if ~mri.moco_intra.nbFirstvols
			nbFirstvols = mri.nt{i_folder};
		else
			nbFirstvols = mri.moco_intra.nbFirstvols;
		end
		disp(['... Number of time series used for averaging: ',num2str(nbFirstvols)])
		fname_data = [mri.nifti.path,folder,mri.nifti.file_data];
		fname_data_firstvols = [mri.nifti.path,folder,mri.nifti.file_data_firstvols];
		cmd = [fsloutput,'fslroi ',fname_data,' ',fname_data_firstvols,' 0 ',num2str(nbFirstvols)];
		j_cprintf('Magenta',[cmd,'\n'])
		j_progress('Extract the time series .......................')
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)

		% Average the first #n time series
		fname_data_firstvols = [mri.nifti.path,folder,mri.nifti.file_data_firstvols];
		fname_data_firstvols_mean = [mri.nifti.path,folder,mri.nifti.file_data_firstvols_mean];
		cmd = [fsloutput,'fslmaths ',fname_data_firstvols,' -Tmean ',fname_data_firstvols_mean];
		j_cprintf('Magenta',[cmd,'\n'])
		j_progress('Average the time series .......................')
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)
		disp(['-> File created: "',mri.nifti.file_data_firstvols_mean,'.nii"'])
	else
		% copy ref image
		j_progress('Copy reference image ..........................')
		fname_data_ref = mri.moco_intra.ref;
		fname_data_firstvols_mean = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data_firstvols_mean];
		copyfile([fname_data_ref],[fname_data_firstvols_mean,'.nii']);
		j_progress(1)
		disp(['... Target image: "',mri.moco_intra.ref,'"'])
	end
	
	% Copy target image at the beggining of the time series
	fname_data = [mri.nifti.path,folder,mri.nifti.file_data];
	fname_data_firstvols_mean = [mri.nifti.path,folder,mri.nifti.file_data_firstvols_mean];
	fname_data_toreg = [mri.nifti.path,folder,'tmp.data_toreg'];
% 	mri.nifti.file_data = 'tmp.data_toreg';
	cmd = [fsloutput,'fslmerge -t ',fname_data_toreg,' ',fname_data_firstvols_mean,' ',fname_data];
	j_cprintf('Magenta',[cmd,'\n'])
	j_progress('Concatenate target image ......................')
	[status result] = unix(cmd); % run UNIX command
	if status, error(result); end % check error
	j_progress(1)
	file_data = 'tmp.data_toreg';
	
end % moco method


% IF USER DID NOT ASK FOR INTRA-RUN MOCO, ONLY APPLY INTER-RUN MOCO
if strcmp(mri.moco_intra.method,'none')
	
	% inter-session registration?
	if mri.moco_session.do
		j_progress('Inter-session registration ....................')
		for i_folder=1:nb_folders

			fname_source = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data];
			fname_moco = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data_moco_intra];
			fname_target = [mri.nifti.path,mri.nifti.folder{i_folder},mri.moco_inter.file_b0];
			cmd = [fsloutput,'flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_moco,' -applyxfm -init ',fname_mat_session];
			[status result] = unix(cmd);
			j_progress(i_folder/nb_folders)
		end
	else
		disp('-> No Inter-session registration')
	end
	j_progress('Create Mean and SD image (for quality check) ..')
	% ... of raw data
	fname_data_merged = [mri.nifti.path,mri.nifti.folder{1},mri.nifti.file_data];
	fname_data_mean = [mri.nifti.path,mri.nifti.folder{1},mri.nifti.file_data,mri.file_suffixe.mean];
	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tmean ',fname_data_mean];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(0.5)
	fname_data_std = [mri.nifti.path,mri.nifti.folder{1},mri.nifti.file_data,mri.file_suffixe.std];
	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tstd ',fname_data_std];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(1)
	nb_folders_moco = 0;
else
% 	nb_folders_moco = nb_folders;
% 	nb_folders_moco = 1;
end

% loop across folders
% for i_folder=1:nb_folders_moco

j_cprintf('-black','FOLDER: %s',folder)
j_cprintf('black','\n')		

if ~strcmp(mri.moco_intra.method,'none')

	% 2d or 3d mode
	switch(mri.moco_intra.method)

	case '2d'
		
		disp(['-> Cost function: ',mri.moco_intra.cost])
		disp(['-> Remove rotation: ',num2str(mri.moco_intra.noRotation)])
		% split data in the Z-dimension
		j_progress('Split data in the Z-dimension .................')
		fname_data = [mri.nifti.path,folder,file_data];
		fname_datasub = [mri.nifti.path,folder,mri.nifti.file_datasub];
		cmd = [fsloutput,'fslsplit ',fname_data,' ',fname_datasub,' -z'];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)
		
		% split the ref image in the Z-dimension
		j_progress('Split the ref image in the Z-dimension ........')
		fname_datasub_ref = [mri.nifti.path,folder,mri.nifti.file_datasub_ref];
		cmd = [fsloutput,'fslsplit ',fname_data_firstvols_mean_crop,' ',fname_datasub_ref,' -z'];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)
				
		% estimate motion for each slice
		j_progress('Estimate motion with mcflirt ..................')
		num = j_numbering(mri.nz{i_folder},4,0);
		for iz=1:mri.nz{1}
			% compute transformation based on the image edges?
			if mri.moco_intra.edge
				fname_datasub = [mri.nifti.path,folder,mri.nifti.file_datasub,num{iz}];
				cmd = [fsloutput,'fslmaths ',fname_datasub,' -edge ',fname_datasub];
				[status result] = unix(cmd); % run UNIX command
				if status, error(result); end % check error
				fname_datasub_ref = [mri.nifti.path,folder,mri.nifti.file_datasub_ref,num{iz}];
				cmd = [fsloutput,'fslmaths ',fname_datasub_ref,' -edge ',fname_datasub_ref];
				[status result] = unix(cmd); % run UNIX command
				if status, error(result); end % check error
			end % edge				
			fname_datasub = [mri.nifti.path,folder,mri.nifti.file_datasub,num{iz}];
			fname_datamoco = [mri.nifti.path,folder,mri.nifti.file_datamoco,num{iz}];
			fname_datasub_ref = [mri.nifti.path,folder,mri.nifti.file_datasub_ref,num{iz}];
			if mri.moco_intra.noRotation
				cmd = [fsloutput,'mcflirt -in ',fname_datasub,' -out ',fname_datamoco,' -r ',fname_datasub_ref,' -rotation 0',' -cost ',mri.moco_intra.cost,' -smooth ',mri.moco_intra.smooth];
			else
				cmd = [fsloutput,'mcflirt -in ',fname_datasub,' -out ',fname_datamoco,' -r ',fname_datasub_ref,' -cost ',mri.moco_intra.cost,' -smooth ',mri.moco_intra.smooth];
			end				
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(iz/mri.nz{1})
		end
	
		% merge moco data
		j_progress('Merge moco data along Z .......................')
		fname_data_moco = [mri.nifti.path,folder,mri.nifti.file_data,mri.nifti.file_data_moco];
		fname_data_moco_3d = [mri.nifti.path,folder,mri.nifti.file_datamoco,'*.nii'];
		cmd = [fsloutput,'fslmerge -z ',fname_data_moco,' ',fname_data_moco_3d];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)
	

	case '3d'
		
		% estimate motion
		fname_data = [mri.nifti.path,folder,file_data];
		fname_datamoco = [mri.nifti.path,folder,'tmp.data_moco'];
		fname_data_ref = [mri.nifti.path,folder,mri.nifti.file_data_firstvols_mean];
		cmd = [fsloutput,'mcflirt -in ',fname_data,' -out ',fname_datamoco,' -r ',fname_data_ref,' -dof ',mri.moco_intra.dof,' -cost ',mri.moco_intra.cost,' -smooth ',mri.moco_intra.smooth,' -report'];
		% specify final interpolation
		if mri.moco_intra.sync_interp
			cmd = [cmd,' -sinc_final -stages 4'];
		end
		% specify rotation status
		if mri.moco_intra.noRotation
			cmd = [cmd,' -rotation 0'];
		end				
		disp(['>> "',cmd,'"'])
		j_progress('Estimate motion with mcflirt ..................')
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)		
		
	case 'afni'
		
		% estimate motion
		disp('Drag/Drop the following command line into a Terminal window. When finished, press a key.')
		fname_data = [mri.nifti.path,folder,file_data,'.nii'];
		fname_datamoco = [mri.nifti.path,folder,'tmp.data_moco.nii'];
		cmd = ['mc-afni2 --i ',fname_data,' --o ',fname_datamoco];
		disp([cmd])
 		pause
% 		j_progress('Estimate motion with AFNI .....................')
% 		[status result] = system(cmd); % run UNIX command
% 		if status, error(result); end % check error
% 		j_progress(1)

	end % switch
	
	% Remove first image of the series (target image)
 	fname_datamoco = [mri.nifti.path,folder,'tmp.data_moco.nii'];
 	fname_datamoco_final = [mri.nifti.path,folder,mri.nifti.file_data,mri.nifti.file_data_moco];
	disp('Remove the first image of the series (target)...')
	cmd = [fsloutput,'fslroi ',fname_datamoco,' ',fname_datamoco_final,' 1 ',num2str(mri.nt{1})];
	j_cprintf('Magenta',[cmd,'\n'])
	[status result] = unix(cmd); % run UNIX command
	if status, error(result); end % check error
% 	j_progress(1)
	disp(['-> File created: "',fname_datamoco_final,'"'])
	
	% Create Mean and STD across time (for quality control)...
	j_progress('Create Mean and SD image (for quality check) ..')
	% ... of moco data
	fname_data_merged = [mri.nifti.path,folder,mri.nifti.file_data,mri.nifti.file_data_moco];
	fname_data_mean = [mri.nifti.path,folder,mri.nifti.file_data,mri.nifti.file_data_moco,mri.file_suffixe.mean];
	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tmean ',fname_data_mean];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(0.25)
	fname_data_std = [mri.nifti.path,folder,mri.nifti.file_data,mri.nifti.file_data_moco,mri.file_suffixe.std];
	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tstd ',fname_data_std];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(0.5)
	% ... of raw data
	fname_data_merged = [mri.nifti.path,folder,mri.nifti.file_data];
	fname_data_mean = [mri.nifti.path,folder,mri.nifti.file_data,mri.file_suffixe.mean];
	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tmean ',fname_data_mean];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(0.75)
	fname_data_std = [mri.nifti.path,folder,mri.nifti.file_data,mri.file_suffixe.std];
	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tstd ',fname_data_std];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(1)

	% Delete temp files
	j_progress('Delete temporary files ........................')
	delete([mri.nifti.path,folder,'tmp.*']);
	j_progress(1)

	% change the default data file name
	mri.nifti.file_data = [mri.nifti.file_data,mri.nifti.file_data_moco];
end





% ====================================================================
% SMOOTH DATA 
% ====================================================================
j_cprintf('black','');
j_cprintf('blue','\nSMOOTH DATA\n');

if mri.smooth.do
 	disp(['-> Smooth size: ',mri.smooth.size])
	
	j_progress('Smooth data ...................................')
	fname_data = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data];
	fname_datasmooth = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data];
	cmd = [fsloutput,'fslmaths ',fname_data,' -s ',mri.smooth.size,' ',fname_datasmooth];
	[status result] = unix(cmd); % run UNIX command
	if status, error(result); end % check error
	j_progress(1)

% 	% change the default data file name
% 	mri.nifti.file_data = mri.nifti.file_data_crop;
else
	fprintf(1,'Skip this step.\n');
end







% =========================================================================
%	CREATE MASK
% =========================================================================
fprintf('')
j_cprintf('blue','\nCREATE MASK\n\n')

if mri.mask.useSameMask
	disp('-> Generate a single mask for all folders')
	nb_folders_mask = 1; % generate a mask from the first folder and copy it into the other folders (in case it is the same sequence, slice prescription, etc.)
else
	disp('-> Generate one mask per folder')
	nb_folders_mask = nb_folders; % generate one mask per folder
end
disp(['-> Mask generation method: "',mri.mask.method,'"'])

% if strcmp(mri.mask.ref,'b0')
i_folder = 1;
fname_mask_ref = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data,mri.file_suffixe.mean];
% elseif strcmp(mri.mask.ref,'dwi')
% 	mask_ref = mri.nifti.file_dwi_mean;
% end
disp(['-> Mask reference: "',fname_mask_ref,'"'])

% loop over folder
for i_folder = 1:nb_folders_mask
	
	j_cprintf('black','\n')
	j_cprintf('-black','FOLDER %i/%i',i_folder,nb_folders_mask)
	j_cprintf('black','\n')

	switch mri.mask.method

	case 'bet' % create mask using BET

		error('TODO')
% 		j_progress('Create mask using BET .........................')
% 		fname_ref = [mri.nifti.path,mri.nifti.folder{i_folder},mask_ref];
% 		fname_mask = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_mask];
% 		cmd = [fsloutput,'bet ',fname_ref,' ',fname_mask,' -f ',num2str(mri.mask.bet_threshold)];
% 		[status result] = unix(cmd);
% 		j_progress(1)
% 
% 		% display mask
% 		if mri.mask.display
% 			reply1 = 'n';
% 			while strcmp(reply1,'n')
% 				mask = read_avw(fname_mask);
% 				j_displayMRI(mask);
% 				reply1 = input('Do you like this mask? y/n [y]: ', 's');
% 				if strcmp(reply1,'n')
% 					txt = ['What threshold would you like? [previous value was ',num2str(mri.mask.bet_threshold),']: '];
% 					reply2 = input(txt);
% 					mri.mask.bet_threshold = reply2;
% 					j_progress('Generate new mask .............................')
% 					fname_ref = [mri.nifti.path,mri.nifti.folder{i_folder},mask_ref];
% 					fname_mask = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_mask];
% 					cmd = [fsloutput,'bet ',fname_ref,' ',fname_mask,' -f ',num2str(mri.mask.bet_threshold)];
% 					[status result] = unix(cmd);
% 					if status, error(result); end
% 					j_progress(1)
% 				end
% 			end
% 			clear reply1 reply2
% 			close % close figure
% 		end % if mri.mask.display

	case 'auto'

		disp('-> Mask generation method: "auto"')
		% display stuff
		fprintf(['Use automatic thresholding method with:\n- FWHM=',num2str(mri.mask.auto.fwhm),'\n- Threshold=',num2str(mri.mask.auto.threshold),'\n'])

		% display mask
		reply1 = 'n';
		while strcmp(reply1,'n')
			% smooth mean DWI
			j_progress('Create mask ...................................')
			fname_ref = fname_mask_ref;
			fname_dwi_smooth = [mri.nifti.path,mri.nifti.folder{i_folder},'tmp.ref_smooth'];
			cmd = [fsloutput,'fslmaths ',fname_ref,' -s ',num2str(mri.mask.auto.fwhm),' ',fname_dwi_smooth];
			[status result] = unix(cmd);
			if status, error(result); end
			% create mask
			cmd = [fsloutput,'fslmaths ',fname_dwi_smooth,' -thr ',num2str(mri.mask.auto.threshold),' -bin ',mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_mask];
			[status result] = unix(cmd);
			if status, error(result); end
			j_progress(1)
			% display mask
			if mri.mask.display
				% load dwi_mean
				dwi_mean = read_avw(fname_ref);
				% load mask
				mask = read_avw([mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_mask]);
				% multiply both images for display purpose
				dwi_mean_masked = dwi_mean.*mask;
				[min_mask index_min_mask] = sort([size(mask,1) size(mask,2) size(mask,3)],'descend');
				dwi_mean_masked = permute(dwi_mean_masked,index_min_mask);
				j_displayMRI(dwi_mean_masked);
				reply1 = input('Do you like this mask? y/n [y]: ', 's');
				if strcmp(reply1,'n')
					txt = ['What FWHM would you like? [previous value was ',num2str(mri.mask.auto.fwhm),']: '];
					reply2 = input(txt);
					mri.mask.auto.fwhm = reply2;
					txt = ['What intensity threshold would you like? [previous value was ',num2str(mri.mask.auto.threshold),']: '];
					reply3 = input(txt);
					mri.mask.auto.threshold = reply3;
				end
				close % close figure
			else
				reply1 = 'y';
			end
		end

		% Delete datasub
		j_progress('Delete temporary files ........................')
		delete([mri.nifti.path,mri.nifti.folder{i_folder},'tmp.*']);
% 		delete([mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_dwi]);
		j_progress(1)

	case 'manual'

		disp('-> Mask generation method: "manual"')
		
		if mri.mask.manual.ask
			% Ask the user to create a mask...
			disp(['** Open a Terminal and go to the following directory: "',mri.nifti.path,mri.nifti.folder{i_folder},'"'])
			disp(['** Then, generate a mask using fslview based on the mean dwi image. To do this, type: "fslview dwi_mean"'])
			disp(['** Once you''re happy with the mask, save it under the name "nodif_brain_mask.nii"'])
			disp(['** Then go back to Matlab and press a key'])
			pause
		end
	end
end % i_folder_mask

% Copy mask in each other folder
if mri.mask.useSameMask
	j_progress('Copy mask in each folder ......................')
	for i_folder = 2:nb_folders
		copyfile([mri.nifti.path,mri.nifti.folder{1},mri.nifti.file_mask,ext],[mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_mask,ext]);
		j_progress(i_folder/nb_folders)
	end
end








% % ====================================================================
% % AVERAGE THE DATA 
% % ====================================================================
% % Average multiple folders
% 
% fprintf('')
% j_cprintf('blue','\nAVERAGE THE DATA\n')
% 
% if mri.average.do
% 	
% 	% Average across time
% 	j_progress('Average data ..................................')
% 	fname_data_merged = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
% 	fname_data_averaged = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_final];
% 	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tmean ',fname_data_averaged];
% 	[status result] = unix(cmd);
% 	if status, error(result); end
% 	j_progress(1)
% 
% 	% Create STD across time (for quality control)
% 	j_progress('Create STD (for quality control) ..............')
% 	fname_data_merged = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
% 	fname_data_std = [mri.nifti.path,mri.nifti.folder_average,'data_std'];
% 	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tstd ',fname_data_std];
% 	[status result] = unix(cmd);
% 	if status, error(result); end
% 	j_progress(1)
% 
% 	% delete temporary data
% % 	j_progress('Delete temporary files ........................')
% % 	delete([fname_data_merged,'.*']);
% % 	j_progress(1)
% 	
% else
% 	
% 	fprintf(1,'Skip this step.\n');
% 
% end







% ====================================================================
% MTR
% ====================================================================
% Compute MTR

if mri.mtr

	fprintf('')
	j_cprintf('blue','\nCompute MTR\n\n')

	% create folder (if does not already exist)
	if ~exist(strcat(mri.nifti.path,mri.nifti.folder_mtr))
		j_progress('Create folder .................................')
		mkdir(strcat(mri.nifti.path,mri.nifti.folder_mtr));
		j_progress(1)
	end
	
	% Split data
	j_progress('Split data ....................................')
	fname_data = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
	fname_t1 = [mri.nifti.path,mri.nifti.folder_mtr,'t1'];
	fname_mt = [mri.nifti.path,mri.nifti.folder_mtr,'mt'];
	cmd = [fsloutput,'fslroi ',fname_data,' ',fname_t1,' 0 1'];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(0.5)
	cmd = [fsloutput,'fslroi ',fname_data,' ',fname_mt,' 1 1'];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(1)
	
	% Average across time
	j_progress('Compute MTR ...................................')
	fname_mtr = [mri.nifti.path,mri.nifti.folder_mtr,mri.nifti.file_mtr];
	cmd = [fsloutput,'fslmaths ',fname_t1,' -sub ',fname_mt,' -div ',fname_t1,' -mul 100 ',fname_mtr];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(1)

end


% save structure
% save([mri.struct.path,mri.struct.file],'mri');


