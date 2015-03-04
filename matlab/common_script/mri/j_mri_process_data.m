% =========================================================================
% FUNCTION
% j_mri_process_data.m
%
% run batch_dcm2nii.m first. It will generate the 'mri' structure.
% 
% COMMENTS
% Julien Cohen-Adad 2010-07-19
% =========================================================================
function mri = j_mri_process_data(mri)


% parameters


% =========================================================================
% START THE SCRIPT
% =========================================================================

% initializations
fsloutput = mri.fsloutput;
nb_folders = length(mri.nifti.folder);
mri.nifti.file_data = mri.nifti.file_data_raw;

j_cprintf('red','\nPROCESS FMRI DATA\n')

j_cprintf('blue','\nREORIENT DATA\n')

% reorient data
if mri.reorient.do
	
	j_progress('Re-orient data according to MNI template ......')
	for i_folder = 1:nb_folders
		% build file names
		fname_data = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data];
		% re-orient data
		cmd = [fsloutput,'fslswapdim ',fname_data,' ',mri.nifti.reorient,' ',fname_data];
		[status result] = unix(cmd);
		j_progress(i_folder/nb_folders)
	end
else
	fprintf(1,'Skip this step.\n');
end

% get data dimensions
disp('Get dimensions of the data...')
for i_folder = 1:nb_folders
	fname_data = [mri.nifti.path,mri.nifti.folder{i_folder},mri.nifti.file_data];
	cmd = ['fslsize ',fname_data];
	[status result] = unix(cmd);
	if status, error(result); end
	dims = j_mri_getDimensions(result);
	mri.nx{i_folder} = dims(1);
	mri.ny{i_folder} = dims(2);
	mri.nz{i_folder} = dims(3);
	mri.nt{i_folder} = dims(4);
	disp(['-> Folder "',mri.nifti.folder{i_folder},'": ',num2str(dims(1)),'x',num2str(dims(2)),'x',num2str(dims(3)),'x',num2str(dims(4)),' (',num2str(i_folder),'/',num2str(nb_folders),')'])
end








% ====================================================================
% CROP DATA 
% ====================================================================
j_cprintf('black','');
j_cprintf('blue','\nCROP DATA\n');

if mri.crop.do
	disp(['-> Crop size: ',mri.crop.size])
	
	j_progress('Crop data .....................................')
	for iFolder=1:nb_folders
		fname_data = [mri.nifti.path,mri.nifti.folder{iFolder},mri.nifti.file_data];
		fname_datacrop = [mri.nifti.path,mri.nifti.folder{iFolder},mri.nifti.file_data_crop];
		cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',mri.crop.size];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(iFolder/nb_folders)
	end

	% change the default data file name
	mri.nifti.file_data = mri.nifti.file_data_crop;
else
	fprintf(1,'Skip this step.\n');
end





% Merge data into 4D file to let the program think that this is actually a
% 4D sequence, so that it can use the "intra-run-moco" solution.
if mri.average.do
	
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
	
	j_progress('Merge data into 4D file .......................')
	fname_data_3d = [];
	for iFolder = 1:nb_folders
		fname_data_3d = strcat(fname_data_3d,[' ',mri.nifti.path,mri.nifti.folder{iFolder},mri.nifti.file_data]);
	end
	fname_data_merged = [mri.nifti.path,mri.nifti.folder_average,'tmp.data_merged'];
	cmd = [fsloutput,'fslmerge -t ',fname_data_merged,' ',fname_data_3d];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(1)

	mri.nifti.file_data = 'tmp.data_merged';
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

%TODO

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

%TODO
	
else
	
	fprintf(1,'Skip this step.\n');

end










% ====================================================================
% INTRA-RUN MOTION CORRECTION 
% ====================================================================
% N.B. Here the motion correction is done on the "averaged" folder. If
% exist...
j_cprintf('black','');
j_cprintf('blue','\nINTRA-RUN MOTION CORRECTION\n');

disp(['-> Motion correction method: "',mri.moco_intra.method,'"'])

switch(mri.moco_intra.method)

case '2d'
	
% 	for i_folder=1:nb_folders
		
%  		j_cprintf('black','\n')
		j_cprintf('-black','FOLDER: %s',mri.nifti.folder_average)
		j_cprintf('black','\n')

		% crop the data
		if ~isempty(mri.moco_intra.crop)
			disp('-> Crop data: YES')
			disp(['-> Crop size: ',mri.moco_intra.crop])
			j_progress('Crop data .....................................')
			fname_data = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
			fname_datacrop = [mri.nifti.path,mri.nifti.folder_average,'tmp.data_crop'];
			cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',mri.moco_intra.crop];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(1)
		else
			disp('-> Crop data: NO')
			fname_datacrop = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
		end
		
		% split data in the Z-dimension
		j_progress('Split data in the Z-dimension .................')
		fname_datasub = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_datasub];
		cmd = [fsloutput,'fslsplit ',fname_datacrop,' ',fname_datasub,' -z'];
		unix(cmd);
		j_progress(1)

		if isempty(mri.moco_intra.ref)
			% Get the first #n time series
			j_progress(['Extract the first #',num2str(mri.moco_intra.nbFirstvols),' time series ..............'])
			fname_data = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
			fname_data_firstvols = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols];
			cmd = [fsloutput,'fslroi ',fname_data,' ',fname_data_firstvols,' 0 ',num2str(mri.moco_intra.nbFirstvols)];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(1)

			% Average the first #n time series
			j_progress(['Average the first #',num2str(mri.moco_intra.nbFirstvols),' time series ..............'])
			fname_data_firstvols = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols];
			fname_data_firstvols_mean = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols_mean];
			cmd = [fsloutput,'fslmaths ',fname_data_firstvols,' -Tmean ',fname_data_firstvols_mean];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(1)
		else
			% copy ref image
			j_progress('Copy reference image ..........................')
			fname_data_ref = mri.moco_intra.ref;
			fname_data_firstvols_mean = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols_mean];
			copyfile([fname_data_ref,'.nii'],[fname_data_firstvols_mean,'.nii']);
			j_progress(1)
			disp(['-> Target image: ',mri.moco_intra.ref])
		end

		% crop the ref image
		if ~isempty(mri.moco_intra.crop)
			j_progress('Crop the ref image ............................')
			fname_data_firstvols_mean = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols_mean];
			fname_data_firstvols_mean_crop = [mri.nifti.path,mri.nifti.folder_average,'tmp.ref_crop'];
			cmd = [fsloutput,'fslroi ',fname_data_firstvols_mean,' ',fname_data_firstvols_mean_crop,' ',mri.moco_intra.crop];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(1)
		else
			fname_data_firstvols_mean_crop = fname_data_firstvols_mean;
		end
		
		% split the ref image in the Z-dimension
		j_progress('Split the ref image in the Z-dimension ........')
		fname_datasub_ref = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_datasub_ref];
		cmd = [fsloutput,'fslsplit ',fname_data_firstvols_mean_crop,' ',fname_datasub_ref,' -z'];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)
				
		% estimate motion for each slice
		disp('Registration:')
		disp(['-> Cost function: ',mri.moco_intra.cost])
		j_progress('Estimate motion with mcflirt ..................')
		num = j_numbering(mri.nz{i_folder},4,0);
		for iz=1:mri.nz{1}
			fname_datasub = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_datasub,num{iz}];
			fname_datamoco = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_datamoco,num{iz}];
			fname_datasub_ref = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_datasub_ref,num{iz}];
			cmd = [fsloutput,'mcflirt -in ',fname_datasub,' -out ',fname_datamoco,' -r ',fname_datasub_ref,' -rotation 0',' -cost ',mri.moco_intra.cost,' -smooth ',mri.moco_intra.smooth];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(iz/mri.nz{1})
		end
		
		% merge moco data
		j_progress('Merge moco data into 4D file ..................')
		fname_data_moco = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_moco_intra];
		fname_data_moco_3d = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_datamoco,'*.nii'];
		cmd = [fsloutput,'fslmerge -z ',fname_data_moco,' ',fname_data_moco_3d];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)

		% merge raw data (for comparison)
		j_progress('Merge raw data into 4D file ...................')
		fname_data_raw = [mri.nifti.path,mri.nifti.folder_average,'data_raw_concat'];
		fname_data_raw_3d = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
		cmd = [fsloutput,'fslmerge -z ',fname_data_raw,' ',fname_data_raw_3d];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)

		% Delete temp files
		j_progress('Delete temporary files ........................')
		delete([mri.nifti.path,mri.nifti.folder_average,'tmp.*']);
		j_progress(1)
% 	end
	
	% change the default data file name
	mri.nifti.file_data = mri.nifti.file_data_moco_intra;


case '3d'
	
% 	for i_folder=1:nb_folders
		
%  		j_cprintf('black','\n')
		j_cprintf('-black','FOLDER: %s',mri.nifti.folder_average)
		j_cprintf('black','\n')

		% crop the data
		if ~isempty(mri.moco_intra.crop)
			disp('-> Crop data: YES')
			disp(['-> Crop size: ',mri.moco_intra.crop])
			j_progress('Crop data .....................................')
			fname_data = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
			fname_datacrop = [mri.nifti.path,mri.nifti.folder_average,'tmp.data_crop'];
			cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',mri.moco_intra.crop];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(1)
		else
			disp('-> Crop data: NO')
			fname_datacrop = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
		end
		
		if isempty(mri.moco_intra.ref)
			% Get the first #n time series
			j_progress(['Extract the first #',num2str(mri.moco_intra.nbFirstvols),' time series ..............'])
			fname_data = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
			fname_data_firstvols = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols];
			cmd = [fsloutput,'fslroi ',fname_data,' ',fname_data_firstvols,' 0 ',num2str(mri.moco_intra.nbFirstvols)];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(1)

			% Average the first #n time series
			j_progress('Average the first #n time series ..............')
			fname_data_firstvols = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols];
			fname_data_firstvols_mean = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols_mean];
			cmd = [fsloutput,'fslmaths ',fname_data_firstvols,' -Tmean ',fname_data_firstvols_mean];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(1)
		else
			% copy ref image
			j_progress('Copy reference image ..........................')
			fname_data_ref = mri.moco_intra.ref;
			fname_data_firstvols_mean = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols_mean];
			copyfile([fname_data_ref,'.nii'],[fname_data_firstvols_mean,'.nii']);
			j_progress(1)
			disp(['-> Target image: ',mri.moco_intra.ref])
		end

		% crop the ref image
		if ~isempty(mri.moco_intra.crop)
			j_progress('Crop the ref image ............................')
			fname_data_firstvols_mean = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols_mean];
			fname_data_firstvols_mean_crop = [mri.nifti.path,mri.nifti.folder_average,'tmp.ref_crop'];
			cmd = [fsloutput,'fslroi ',fname_data_firstvols_mean,' ',fname_data_firstvols_mean_crop,' ',mri.moco_intra.crop];
			[status result] = unix(cmd); % run UNIX command
			if status, error(result); end % check error
			j_progress(1)
		else
			fname_data_firstvols_mean_crop = fname_data_firstvols_mean;
		end
		
		% estimate motion
		disp('Registration:')
		disp(['-> Cost function: ',mri.moco_intra.cost])
		j_progress('Estimate motion with mcflirt ..................')
		fname_datasub = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
		fname_datamoco = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_moco_intra];
		fname_datasub_ref = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_firstvols_mean];
		cmd = [fsloutput,'mcflirt -in ',fname_datasub,' -out ',fname_datamoco,' -r ',fname_datasub_ref,' -dof ',mri.moco_intra.dof,' -cost ',mri.moco_intra.cost,' -smooth ',mri.moco_intra.smooth,' -report'];
		[status result] = unix(cmd); % run UNIX command
		if status, error(result); end % check error
		j_progress(1)
		
		% merge raw data (for comparison)
		j_progress('Merge raw data into 4D file ...................')
		fname_data_raw = [mri.nifti.path,mri.nifti.folder_average,'data_raw_concat'];
		fname_data_raw_3d = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
		copyfile([fname_data_raw_3d,'.nii'],[fname_data_raw,'.nii']);
		j_progress(1)
		
		% Delete temp files
		j_progress('Delete temporary files ........................')
		delete([mri.nifti.path,mri.nifti.folder_average,'tmp.*']);
		j_progress(1)
% 	end
	
	% change the default data file name
	mri.nifti.file_data = mri.nifti.file_data_moco_intra;

	
% No intra-run motion correction
case 'none'

	% inter-session registration
	if mri.moco_session.do
		
		j_progress('Inter-session registration ....................')
		for i_folder=1:nb_folders

			fname_source = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
			fname_moco = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_moco_intra];
			fname_target = [mri.nifti.path,mri.nifti.folder_average,mri.moco_inter.file_b0];
			cmd = [fsloutput,'flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_moco,' -applyxfm -init ',fname_mat_session];
			[status result] = unix(cmd);
			j_progress(i_folder/nb_folders)
		end
	else
		fprintf(1,'\nSkip this step.\n');
	end
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




% ====================================================================
% AVERAGE THE DATA 
% ====================================================================
% Average multiple folders
% N.B. Do that direction-by-direction to avoid 'Out of memory'

fprintf('')
j_cprintf('blue','\nAVERAGE THE DATA\n\n')

if mri.average.do
	
	% Average across time
	j_progress('Average data ..................................')
	fname_data_merged = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
	fname_data_averaged = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data_final];
	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tmean ',fname_data_averaged];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(1)

	% Create STD across time (for quality control)
	j_progress('Create STD (for quality control) ..............')
	fname_data_merged = [mri.nifti.path,mri.nifti.folder_average,mri.nifti.file_data];
	fname_data_std = [mri.nifti.path,mri.nifti.folder_average,'data_std'];
	cmd = [fsloutput,'fslmaths ',fname_data_merged,' -Tstd ',fname_data_std];
	[status result] = unix(cmd);
	if status, error(result); end
	j_progress(1)

	% delete temporary data
% 	j_progress('Delete temporary files ........................')
% 	delete([fname_data_merged,'.*']);
% 	j_progress(1)
	
else
	
	fprintf(1,'Skip this step.\n');

end

% save structure
save([mri.struct.path,mri.struct.file],'mri');


