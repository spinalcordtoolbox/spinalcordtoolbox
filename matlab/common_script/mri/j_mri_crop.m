% =========================================================================
% FUNCTION
% j_mri_crop.m
%
% Shrink MR image file by means of a mask. By default, cropped data is
% written with prefixe "crop_".
% The mask must:
% - either be a parallelipiped or having the same cropping area per slice
% - have the same size as the data
% 
% INPUT
% fname_data		full file name of data to crop
% fname_mask		full file name of cropping mask.
%
% OUTPUT
% (-)
% 
% COMMENTS
% Julien Cohen-Adad 2010-04-10
% =========================================================================
function j_mri_crop(fname_data,fname_mask)


% default initialization
prefixe_write		= 'crop_';
if (nargin<2), help j_mri_crop; return; end

% load data
j_progress('Load data .....................................')
[data,dims,scales,bpp,endian] = read_avw(fname_data);
nx = dims(1);
ny = dims(2);
nz = dims(3);
j_progress(1)

% Load mask
j_progress('Load mask .....................................')
[mask,dims,scales,bpp,endian] = read_avw(fname_mask);
j_progress(1)

% Crop the data
data_crop = [];
j_progress('Crop the data .................................')
for iz=1:nz
    [x y] = find(mask(:,:,iz));
    if ~isempty(x) & ~isempty(y)
        xmin = min(x);
        xmax = max(x);
        ymin = min(y);
        ymax = max(y);
		data_crop(:,:,iz) = data(xmin:xmax,ymin:ymax,iz);
% 		
%         z(i) = i;
%         index_zmin = min(find(z));
%         index_zmax = max(find(z));
    end
end
j_progress(1)

% Write data
j_progress('Write cropped data ..........................')
[path_read,file_read] = fileparts(fname_data);
fname_write = [path_read,prefixe_write,file_read];
save_avw(data_crop,fname_write,'d',scales);
j_progress(1)

disp('done.')



% =========================================================================
% SCRIPT
% j_dmri_process_data.m
%
% run batch_dcm2nii.m first.
% 
% COMMENTS
% Julien Cohen-Adad 2009-09-25
% =========================================================================
function dmri = j_dmri_process_data(dmri)

% load dmri

% parameters
% dmri.nifti.file_data	= dmri.nifti.file_data_raw;
% dmri.nifti.file_bvals	= dmri.nifti.file_bvals_raw;
% dmri.nifti.file_bvecs	= dmri.nifti.file_bvecs_raw;


% =========================================================================
% START THE SCRIPT
% =========================================================================

% initializations
ext = '.nii';
flsoutput = 'export FSLOUTPUTTYPE=NIFTI; ';
% fname_mask = [dmri.nifti.path,dmri.nifti.file_mask];
nb_folders = length(dmri.nifti.folder);
nb_b0 = dmri.nifti.nb_b0;
nb_dir = dmri.nifti.nb_dir;
dmri.nifti.file_data = dmri.nifti.file_data_raw;
dmri.nifti.file_bvecs = dmri.nifti.file_bvecs_raw;
dmri.nifti.file_bvals = dmri.nifti.file_bvals_raw;

% file_data = dmri.nifti.file_data;
	
j_cprintf('red','\nPROCESS DIFFUSION DATA\n')


% % find the input data
% j_progress('Find the input data .....................................')
% fname = [dmri.path,filesep,dmri.file_data];
% if ~exist(fname)
% 	disp('Data not found. Please check your file name.')
% end
% j_progress(1)

j_cprintf('blue','\nREORIENT DATA\n\n')

% reorient data
if dmri.reorient.do
	
	j_progress('Re-orient data according to MNI template ......')
	for i_folder = 1:nb_folders
		% build file names
		fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
		% re-orient data
		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslswapdim ',fname_data,' ',dmri.nifti.reorient,' ',fname_data];
		[status result] = unix(cmd);
		j_progress(i_folder/nb_folders)
	end
else
	fprintf(1,'Skip this step.\n');
end


% get data dimensions
j_progress('Get dimensions of the data ....................')
for i_folder = 1:nb_folders
	fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
	[img,dims,scales,bpp,endian] = read_avw(fname_data);
	dmri.nx{i_folder} = dims(1);
	dmri.ny{i_folder} = dims(2);
	dmri.nz{i_folder} = dims(3);
	dmri.nt{i_folder} = dims(4);
	j_progress(i_folder/nb_folders)
end
clear img




% ====================================================================
% INTER-RUN MOTION CORRECTION 
% ====================================================================
% Perform inter-run motion correction, by registering the first b=0 found
% in runs #2 -> #last to the first b=0 found in run #1.
% For each folder, write a mat file that will be used during the intra-run
% motion correction (as an initialization matrix).
j_cprintf('blue','\nINTER-RUN MOTION CORRECTION\n\n')

if dmri.moco_inter.do

	% find where are the B0
	index_b0 = find(dmri.gradients.bvals==0);

	% Extract the first B0 in each folder
	j_progress('Extract the first b0 in each folder ...........')
	for i_folder=1:nb_folders
		fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
		fname_first_b0 = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.moco_inter.file_b0];
		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslroi ',fname_data,' ',fname_first_b0,' ',num2str(index_b0(1)-1),' 1'];
		unix(cmd);
		j_progress(i_folder/nb_folders)
	end
	
	% Estimate subject motion
	j_progress('Estimate inter-run motion .....................')
	fname_target = [dmri.nifti.path,dmri.nifti.folder{1},dmri.moco_inter.file_b0];
	fname_mat_inter = {};
	for i_folder=2:nb_folders
		fname_source = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.moco_inter.file_b0];
		fname_moco_inter = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.moco_inter.file_moco];
		fname_mat_inter{i_folder} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.moco_inter.file_mat];
		cmd = ['flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_moco_inter,' -omat ',fname_mat_inter{i_folder},' -cost ',dmri.moco_inter.cost];
		[status result] = unix(cmd);
		j_progress((i_folder-1)/(nb_folders-1))
	end
	
	% generate the identity matrix for the first run of data
	fname_mat_inter{1} = [dmri.nifti.path,dmri.nifti.folder{1},dmri.moco_inter.file_mat];
	fid = fopen(fname_mat_inter{1},'w');
	fprintf(fid,'%i %i %i %i\n',[1 0 0 0]);
	fprintf(fid,'%i %i %i %i\n',[0 1 0 0]);
	fprintf(fid,'%i %i %i %i\n',[0 0 1 0]);
	fprintf(fid,'%i %i %i %i\n',[0 0 0 1]);
	fclose(fid);

% 	
% 	j_progress('Correct motion in all DW images ...............')
% 	fname_target = [dmri.nifti.path,dmri.nifti.folder{1},dmri.nifti.file_nodif_mean];
% 	for i=2:nb_folders
% 		fname_source = [dmri.nifti.path,dmri.nifti.folder{i},dmri.nifti.file_data];
% 		fname_moco = [dmri.nifti.path,dmri.nifti.folder{i},dmri.nifti.file_data_moco];
% 		fname_mat = [dmri.nifti.path,dmri.nifti.folder{i},dmri.nifti.file_moco_mat];
% 		cmd = ['flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_moco,' -applyxfm -init ',fname_mat];
% 		[status result] = unix(cmd);
% 		j_progress(i/nb_folders)
% 	end
% 	
	% MAKE THE OUTPUT IN NII INSTEAD OF NII.GZ
% 	
% 	% copy file name for first folder
% 	copyfile([dmri.nifti.path,dmri.nifti.folder{1},dmri.nifti.file_data],[dmri.nifti.path,dmri.nifti.folder{1},'data_moco.nii']);
% 	% and zip it
% 	gzip([dmri.nifti.path,dmri.nifti.folder{1},'data_moco.nii']);
% 	% and delete the other stuff
% 	delete([dmri.nifti.path,dmri.nifti.folder{1},'data_moco.nii'])
% 	
% 	% SHOULD APPLY THE IDENTITY TRANSFO - INSTEAD OF COPYING - TO THE FIRST
% 	% FOLDER DUE TO INTERPOLATION
% 	% change the default data name in the structure
% 	dmri.nifti.file_data = dmri.nifti.file_data_moco;
% 
else
	
	fprintf(1,'Skip this step.\n');

end










% ====================================================================
% INTER-SESSION MOTION CORRECTION 
% ====================================================================
% Perform inter-session motion correction, by registering the first b=0 of
% run #1 into the b=0 image specified by the user.
% For each folder, write a mat file that will be used during the intra-run
% motion correction (as an initialization matrix).
j_cprintf('blue','\nINTER-SESSION MOTION CORRECTION\n\n')

if dmri.moco_session.do

	% find where are the b0
	index_b0 = find(dmri.gradients.bvals==0);

	% Extract the first b0 of the first folder
	j_progress('Extract the first b0 of the first folder ......')
	fname_data = [dmri.nifti.path,dmri.nifti.folder{1},dmri.nifti.file_data];
	fname_first_b0 = [dmri.nifti.path,dmri.nifti.folder{1},dmri.moco_inter.file_b0];
	cmd = ['export FSLOUTPUTTYPE=NIFTI; fslroi ',fname_data,' ',fname_first_b0,' ',num2str(index_b0(1)-1),' 1'];
	unix(cmd);
	j_progress(1)
	
	% Estimate subject motion between sessions
	j_progress('Estimate inter-session motion .................')
	fname_target = [dmri.moco_session.fname];
	fname_source = [dmri.nifti.path,dmri.nifti.folder{1},dmri.moco_inter.file_b0];
	fname_moco_session = [dmri.nifti.path,dmri.nifti.folder{1},dmri.moco_session.file_moco];
	fname_mat_session = [dmri.nifti.path,dmri.nifti.folder{1},dmri.moco_session.file_mat];
	cmd = ['flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_moco_session,' -omat ',fname_mat_session,' -cost ',dmri.moco_session.cost,' -dof 6'];
	[status result] = unix(cmd);
	j_progress(1)
	
else
	
	fprintf(1,'Skip this step.\n');

end













% ====================================================================
% INDEXATION JOB
% ====================================================================

j_cprintf('black','');
j_cprintf('blue','\nINDEXATION JOB\n');
for i_folder = 1:nb_folders
	
 		j_cprintf('black','\n')
		j_cprintf('-black','RUN %i/%i',i_folder,nb_folders)
		j_cprintf('black','\n')

		% split the 4D dataset into 3D
		j_progress('Split the 4D dataset into 3D ..................')
		fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
		fname_datasub = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub];
		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslsplit ',fname_data,' ',fname_datasub];
		unix(cmd);
		j_progress(1)
		
		% Find where are the b0
		index_b0 = find(dmri.gradients.bvals<=dmri.b0);
		nb_subs = length(index_b0);
		
		% find which files are the b0
		j_progress('Find which files are the b=0 images ...........')
		nb_files = length(dmri.gradients.bvals);
		num = j_numbering(nb_files,4,0);
		for i_sub = 1:nb_subs
			fname_datasub_b0{i_folder,i_sub} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub,num{index_b0(i_sub)}];
		end
		j_progress(1)

		% Merge b0 images
		j_progress('Merge b0 images ...............................')
		fname_nodif{i_folder} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra];
		cmd_raw = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_nodif{i_folder}];
		for i_sub = 1:nb_subs
			% raw b0 images
			cmd_raw = strcat(cmd_raw,[' ',fname_datasub_b0{i_folder,i_sub},'.*']);
			% moco b0 images
% 			fname_b0_moco{i_sub} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco,num{index_b0(i_sub)},'.*'];
% 			cmd_moco = strcat(cmd_moco,[' ',fname_b0_moco{i_sub}]);
		end
		[status result] = unix(cmd_raw);
% 		[status result] = unix(cmd_moco);
		j_progress(1)

		% Average b0
		j_progress('Average b0 images .............................')
		fname_b0_mean = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra_mean];
		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmaths ',fname_nodif{i_folder},' -Tmean ',fname_b0_mean];
		[status result] = unix(cmd);
		j_progress(1)
		
		% Find where are the DWI
		j_progress('Find where the DWI are ........................')
		bvals = j_readFile([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvals]);
		index_dwi = find(bvals>dmri.b0);
		nb_diff = length(index_dwi);
		j_progress(1)
	
		% find which files are the DWI
		j_progress('Find which files are the DWI ..................')
		nb_files = length(dmri.gradients.bvals);
		num = j_numbering(nb_files,4,0);
		fname_datasub_dwi = {};
		for iDiff = 1:nb_diff
			fname_datasub_dwi{iDiff} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub,num{index_dwi(iDiff)}];
		end
		j_progress(1)
		
		% Merge DW images
		j_progress('Merge DW images ...............................')
		fname_dwi = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_dwi];
		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_dwi];
		for iDiff = 1:nb_diff
			cmd = strcat(cmd,[' ',fname_datasub_dwi{iDiff}]);
		end
		[status result] = unix(cmd);
		j_progress(1)

		% Average DWI
		j_progress('Average DW images .............................')
		fname_dwi_mean{i_folder} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_dwi_mean];
		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmaths ',fname_dwi,' -Tmean ',fname_dwi_mean{i_folder}];
		[status result] = unix(cmd);
		j_progress(1)
end











% ====================================================================
% INTRA-RUN MOTION CORRECTION 
% ====================================================================
j_cprintf('black','');
j_cprintf('blue','\nINTRA-RUN MOTION CORRECTION\n');

if ~strcmp(dmri.moco_intra.method,'none')

	for i_folder=1:nb_folders
		
 		j_cprintf('black','\n')
		j_cprintf('-black','RUN %i/%i',i_folder,nb_folders)
		j_cprintf('black','\n')

		% create initialization motion correction matrix (which may further be modified if there is any inter-session registration)
		fname_mat_intra_init = [dmri.nifti.path,dmri.nifti.folder{i_folder},'mat_intra'];
		fid = fopen(fname_mat_intra_init,'w');
		fprintf(fid,'%f %f %f %f\n',[1 0 0 0]);
		fprintf(fid,'%f %f %f %f\n',[0 1 0 0]);
		fprintf(fid,'%f %f %f %f\n',[0 0 1 0]);
		fprintf(fid,'%f %f %f %f\n',[0 0 0 1]);
		fclose(fid);

		% If there is inter-run motion corection
		if dmri.moco_inter.do
			j_progress('Update matricies to account inter-run motion ..')
			% load intra-reg matrix
			mat_intra = textread(fname_mat_intra_init);
			R_intra = mat_intra(1:3,1:3);
			T_intra = mat_intra(1:3,4);
			% load inter-reg matrix
			mat_inter = textread(fname_mat_inter{i_folder});
			R_inter = mat_inter(1:3,1:3);
			T_inter = mat_inter(1:3,4);
			% multiply rotation matrices
			R_total = R_intra*R_inter;
			% add translation matrices
			T_total = T_intra+T_inter;
			% write new matrix (overwrite the old one)				
			mat_total = cat(2,R_total,T_total);
			mat_total(4,:) = [0 0 0 1];
			fid = fopen(fname_mat_intra_init,'w');
			fprintf(fid,'%f %f %f %f\n',mat_total(1,:));
			fprintf(fid,'%f %f %f %f\n',mat_total(2,:));
			fprintf(fid,'%f %f %f %f\n',mat_total(3,:));
			fprintf(fid,'%f %f %f %f\n',mat_total(4,:));
			fclose(fid);
			j_progress(1)
		end

		% If there is inter-session registration
		if dmri.moco_session.do
			j_progress('Update matricies for inter-session registration')
			% load intra-reg matrix
			mat_intra = textread(fname_mat_intra_init);
			R_intra = mat_intra(1:3,1:3);
			T_intra = mat_intra(1:3,4);
			% load inter-reg matrix
			mat_session = textread(fname_mat_session);
			R_session = mat_session(1:3,1:3);
			T_session = mat_session(1:3,4);
			% multiply rotation matrices
			R_total = R_intra*R_session;
			% add translation matrices
			T_total = T_intra+T_session;
			% write new matrix (overwrite the old one)				
			mat_total = cat(2,R_total,T_total);
			mat_total(4,:) = [0 0 0 1];
			fid = fopen(fname_mat_intra_init,'w');
			fprintf(fid,'%f %f %f %f\n',mat_total(1,:));
			fprintf(fid,'%f %f %f %f\n',mat_total(2,:));
			fprintf(fid,'%f %f %f %f\n',mat_total(3,:));
			fprintf(fid,'%f %f %f %f\n',mat_total(4,:));
			fclose(fid);
			j_progress(1)
		end
			
		% switch between moco methods
		disp(['-> Motion correction method: "',dmri.moco_intra.method,'"'])
		switch dmri.moco_intra.method
	
		case 'b0'
			
			% assign the first motion correction matrix as being the
			% inter-run and inter-session registration matrix
			fname_mat_intra{1} = fname_mat_intra_init;
			
			% estimate motion for each b0
			j_progress('Estimate motion for each b0 ...................')
			fname_moco_intra = 'temp';
			fname_target = fname_datasub_b0{1}; % target is the first b0
% 			fname_mat_intra = [];
			for i_sub = 2:nb_subs
				fname_mat_intra{i_sub} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_moco_intra_mat,num2str(i_sub)];
				cmd = ['flirt -in ',fname_datasub_b0{i_folder,i_sub},' -ref ',fname_target,' -out ',fname_moco_intra,' -omat ',fname_mat_intra{i_sub},' -cost ',dmri.moco_intra.cost,' -init ',fname_mat_intra_init];
				[status result] = unix(cmd);
				j_progress(i_sub/nb_subs)
			end

			% Correct motion on DWI data
			j_progress('Correct motion on DWI data ....................')
			fname_target = fname_datasub_b0{1}; % target is the first b0
			i_sub = 1; % initialize i_sub index
			index_b0(nb_subs+1) = 10000;
			num = j_numbering(nb_files,4,0);
			for i_file = 1:nb_files
				fname_source = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub,num{i_file}];
				fname_moco{i_file} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco,num{i_file}];
				% eventually update the i_sub index
				if (i_file==index_b0(i_sub+1))
					i_sub = i_sub+1;
				end
				cmd = ['export FSLOUTPUTTYPE=NIFTI; flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_moco{i_file},' -applyxfm -init ',fname_mat_intra{i_sub}];
				[status result] = unix(cmd);
				j_progress(i_file/nb_files)
			end

	

		% motion correction of DWI + b=0 based on the mean DWI
		case 'dwi'

			% TODO: register the mean DWI over the first b=0 image and
			% apply the correction matrix to the mat_init matrix
			
			% volume-based or slice-by-slice motion correction?
			if ~dmri.moco_intra.slicewise
				disp('-> Volume-based motion correction')
				
				% estimate motion for all volumes (including b0)
				j_progress('Estimate motion with mcflirt ..................')
				fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
				fname_temp = [dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp_data'];
				cmd = ['export FSLOUTPUTTYPE=NIFTI; mcflirt -in ',fname_data,' -refvol ',num2str(index_dwi(1)-1),' -out ',fname_temp,' -init ',fname_mat_intra_init];
				[status result] = unix(cmd);
				j_progress(1)
				
			else % slice-by-slice motion correction
				disp('-> Slice-by-slice motion correction')
				
				% split the data into Z dimension
				j_progress('Split the data into Z dimension ...............')
				fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
				fname_data_splitZ = [dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp_data_splitZ'];
				cmd = ['export FSLOUTPUTTYPE=NIFTI; fslsplit ',fname_data,' ',fname_data_splitZ,' -z'];
				[status result] = unix(cmd);
				j_progress(1)

				% split the reference data into Z dimension
				j_progress('Split the reference data into Z dimension .....')
				fname_data_ref = [fname_dwi_mean{i_folder}];
				fname_data_ref_splitZ = ['tmp_dwi_splitZ'];
				cmd = ['export FSLOUTPUTTYPE=NIFTI; fslsplit ',fname_data_ref,' ',fname_data_ref_splitZ,' -z'];
				[status result] = unix(cmd);
				j_progress(1)

				% estimate motion on each slice individually
				for iZ = 1:dmri.nz{i_folder}
					disp(['-> Slice ',num2str(iZ),'/',num2str(dmri.nz{i_folder}),':'])
					
					% split into T dimension
					j_progress('Split into T dimension ........................')
					fname_data_splitZ_num{iZ} = [dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp_data_splitZ',num{iZ}];
					fname_data_splitZ_splitT = [fname_data_splitZ_num{iZ},'_splitT'];
					cmd = ['export FSLOUTPUTTYPE=NIFTI; fslsplit ',fname_data_splitZ_num{iZ},' ',fname_data_splitZ_splitT];
					[status result] = unix(cmd);
					j_progress(1)
					
					% do motion correction
					fname_data_ref_splitZ_num{iZ} = [fname_data_ref_splitZ,num{iZ}];
					j_progress('Do motion correction ..........................')
					for iT = 1:dmri.nt{i_folder}
						% if b=0 image, do nothing => at the end we have
						% only corrected for the DW images.
						if isempty(find(index_b0==iT))
							fname_data_splitZ_splitT_num{iZ,iT} = [fname_data_splitZ_splitT,num{iT}];
							fname_data_splitZ_splitT_moco_num{iZ,iT} = [fname_data_splitZ_splitT_num{iZ,iT},'_moco'];
							cmd = ['export FSLOUTPUTTYPE=NIFTI; flirt -in ',fname_data_splitZ_splitT_num{iZ,iT},' -ref ',fname_data_ref_splitZ_num{iZ},' -out ',fname_data_splitZ_splitT_moco_num{iZ,iT},' -2D -init ',fname_mat_intra_init];
							[status result] = unix(cmd);
						end
						j_progress(iT/dmri.nt{i_folder})
					end
					
				end	% iZ
				
				% merge into Z dimension
				j_progress('Merge back into Z dimension ...................')
				for iT = 1:dmri.nt{i_folder}
					fname_data_splitT_moco_num{iT} = [dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp_data_splitT',num{iT},'_moco'];
					cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -z ',fname_data_splitT_moco_num{iT}];
					for iZ = 1:dmri.nz{i_folder}
						cmd = strcat(cmd,[' ',fname_data_splitZ_splitT_moco_num{iZ,iT}]);
					end
					[status result] = unix(cmd);
					j_progress(iT/dmri.nt{i_folder})
				end
				% -> create the files: tmp_data_splitT*.nii
				
				
% 				% merge into T dimension
% 				j_progress('Merge back into T dimension ...................')
% 				fname_data_moco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_moco_intra];
% 				cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_data_moco];
% 				for iT = 1:dmri.nt
% 					cmd = strcat(cmd,[' ',fname_data_splitT_moco_num{iT}]);
% 				end
% 				[status result] = unix(cmd);
% 				j_progress(1)
				
			end % if ~dmri.moco_intra.slicewise
			
		end % switch dmri.moco_intra.method	
		
		% Merge b0 images
% 		j_progress('Merge b0 images ...............................')
% 		fname_nodif{i_folder} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra];
% 		fname_nodif_moco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra_moco];
% 		cmd_raw = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_nodif{i_folder}];
% 		cmd_moco = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_nodif_moco];
% 		for i_sub = 1:nb_subs
% 			% raw b0 images
% 			cmd_raw = strcat(cmd_raw,[' ',fname_datasub_b0{i_folder,i_sub},'.*']);
% 			% moco b0 images
% 			fname_b0_moco{i_sub} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco,num{index_b0(i_sub)},'.*'];
% 			cmd_moco = strcat(cmd_moco,[' ',fname_b0_moco{i_sub}]);
% 		end
% 		[status result] = unix(cmd_raw);
% % 		[status result] = unix(cmd_moco);
% 		j_progress(1)
% 
% 		% Average b0
% 		j_progress('Average b0 images .............................')
% 		fname_b0_mean = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra_mean];
% 		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmaths ',fname_nodif{i_folder},' -Tmean ',fname_b0_mean];
% 		[status result] = unix(cmd);
% 		j_progress(1)
% 
% 		% Delete b0
% 		j_progress('Delete b0 images ..............................')
% 		for i_sub = 1:nb_subs
% 			delete(fname_b0_moco{i_sub});
% 			j_progress(i_sub/nb_subs)
% 		end
		
		% do you want a mean b=0 image at the beggining of the serie?
		j_progress('Add mean b0 as the first volume ...............')
		if dmri.moco_intra.meanb0
			disp('-> Average b=0 images')
			fname_moco{1} = [dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp_data_splitT0000'];
			copyfile([fname_b0_mean,ext],[fname_moco{1},ext]);
			j_progress(1)
			% make clean bvals/bvecs files (remove the b0 inside)
			disp('-> Make clean bvals/bvecs files')
			bvecs = textread([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvecs]);
			bvals = textread([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvals]);
			index_b0(nb_subs+1) = 10000;
			i_sub = 2;
			i_file_new = 1;
			bvecs_new = [];
			bvals_new = [];
			for i_file = 1:nb_files
				if i_file~=index_b0(i_sub)
					bvecs_new(i_file_new,:) = bvecs(i_file,:);
					bvals_new(i_file_new,:) = bvals(i_file,:);
					i_file_new = i_file_new+1;
				else
					i_sub = i_sub+1;
				end
			end
		else % Do not average b=0 images
			fname_moco{1} = [dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp_data_splitT0000'];
			copyfile([fname_nodif{i_folder},ext],[fname_moco{1},ext]);
			j_progress(1)
			disp('-> Do not average b=0 images')
			% make clean bvals/bvecs files (remove the b0 inside)
			bvecs_new = textread([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvecs]);
			bvals_new = textread([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvals]);
		end

		% merge data
		j_progress('Merge data into 4D file .......................')
		fname_data_moco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_moco_intra];
		fname_data_moco_3d = [dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp_data_splitT*.nii'];
		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_data_moco,' ',fname_data_moco_3d];
		[status result] = unix(cmd);
		if status, error(result); end
		j_progress(1)
		
		% update structure
		dmri.gradients.bvecs = bvecs_new;
		dmri.gradients.bvals = bvals_new;
		% TODO: FIND A WAY TO CORRECT B-MATRIX DUE TO SUBJECT MOTION
		% create new bvecs/bvals files
		fid_bvecs = fopen([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvecs_moco_intra],'w');
		fid_bvals = fopen([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvals_moco_intra],'w');
		for i_dir=1:size(bvecs_new,1)
			fprintf(fid_bvecs,'%f %f %f\n',bvecs_new(i_dir,:));
			fprintf(fid_bvals,'%i\n',bvals_new(i_dir));
		end
		fclose(fid_bvecs);
		fclose(fid_bvals);

		% Delete temp files
		j_progress('Delete temporary files ........................')
		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub,'*.*']);
		j_progress(0.25)
		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},'datamoco*.*']);
		j_progress(0.50)
		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},'moco_intra_mat*']);
		j_progress(0.75)
		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},'moco_inter_mat*']);
		j_progress(0.90)
 		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp_*.*']);
		j_progress(1)

		if dmri.delete_raw_data
			j_progress('Delete original data ..........................')
			delete([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_raw]);
			j_progress(1)
		end

	end % i_folder
	
	% change the default data file name
	dmri.nifti.file_data = dmri.nifti.file_data_moco_intra;
	dmri.nifti.file_bvecs = dmri.nifti.file_bvecs_moco_intra;
	dmri.nifti.file_bvals = dmri.nifti.file_bvals_moco_intra;
		

% No intra-run motion correction
else

	% inter-session registration
	if dmri.moco_session.do
		
		j_progress('Inter-session registration ....................')
		for i_folder=1:nb_folders

			fname_source = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
			fname_moco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_moco_intra];
			fname_target = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.moco_inter.file_b0];
			cmd = ['export FSLOUTPUTTYPE=NIFTI; flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_moco,' -applyxfm -init ',fname_mat_session];
			[status result] = unix(cmd);
			j_progress(i_folder/nb_folders)
		end
	else
		fprintf(1,'\nSkip this step.\n');
	end
end

% save structure
save([dmri.struct.path,dmri.struct.file],'dmri');















% =========================================================================
%	CREATE MASK
% =========================================================================
fprintf('')
j_cprintf('blue','\nCREATE MASK\n\n')

if dmri.mask.useSameMask
	disp('-> Generate a single mask for all folders')
	nb_folders_mask = 1; % generate a mask from the first folder and copy it into the other folders (in case it is the same sequence, slice prescription, etc.)
else
	disp('-> Generate one mask per folder')
	nb_folders_mask = nb_folders; % generate one mask per folder
end

% loop over folder
for i_folder = 1:nb_folders_mask
	
	j_cprintf('black','\n')
	j_cprintf('-black','RUN %i/%i',i_folder,nb_folders)
	j_cprintf('black','\n')

	switch dmri.mask.method

	case 'bet' % create mask using BET

		disp('-> Mask generation method: "bet"')
		j_progress('Create mask using BET .........................')
		fname_b0_mean = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra_mean];
		fname_mask = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_mask];
		cmd = [flsoutput,'bet ',fname_b0_mean,' ',fname_mask,' -m -f ',num2str(dmri.mask.bet_threshold)];
		[status result] = unix(cmd);
		j_progress(1)

		% display mask
		if dmri.mask.display
			reply1 = 'n';
			while strcmp(reply1,'n')
				mask = j_mri_read(fname_mask);
				j_displayMRI(mask);
				reply1 = input('Do you like this mask? y/n [y]: ', 's');
				if strcmp(reply1,'n')
					txt = ['What threshold would you like? [previous value was ',num2str(dmri.mask.bet_threshold),']: '];
					reply2 = input(txt);
					dmri.mask.bet_threshold = reply2;
					j_progress('Generate new mask .............................')
					fname_b0_mean = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra_mean];
					fname_mask = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_mask];
					cmd = ['bet ',fname_b0_mean,' ',fname_mask,' -m -f ',num2str(dmri.mask.bet_threshold)];
					[status result] = unix(cmd);
					j_progress(1)
				end
			end
			clear reply1 reply2
		% Copy mask in each other folder
		j_progress('Copy mask in each folder ......................')
		for i_folder = 2:nb_folders
			copyfile([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_mask],[dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_mask]);
			j_progress(i_folder/nb_folders)
		end
		if nb_folders==1, j_progress(1); end

		end

	case 'auto'

		disp('-> Mask generation method: "auto"')
		% display stuff
		fprintf(['Use automatic thresholding method with:\n- FWHM=',num2str(dmri.mask.auto.fwhm),'\n- Threshold=',num2str(dmri.mask.auto.threshold),'\n'])

		% display mask
		reply1 = 'n';
		while strcmp(reply1,'n')
			% smooth mean DWI
			j_progress('Create mask ...................................')
			fname_dwi_smooth = [dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp_dwi_smooth'];
			cmd = [flsoutput,'fslmaths ',fname_dwi_mean{i_folder},' -s ',num2str(dmri.mask.auto.fwhm),' ',fname_dwi_smooth];
			[status result] = unix(cmd);
			% create mask
			cmd = [flsoutput,'fslmaths ',fname_dwi_smooth,' -thr ',num2str(dmri.mask.auto.threshold),' -bin ',dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_mask];
			[status result] = unix(cmd);
			j_progress(1)
			% display mask
			if dmri.mask.display
				% load dwi_mean
				dwi_mean = read_avw(fname_dwi_mean{i_folder});
				% load mask
				mask = read_avw([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_mask]);
				% multiply both images for display purpose
				dwi_mean_masked = dwi_mean.*mask;
				[min_mask index_min_mask] = sort([size(mask,1) size(mask,2) size(mask,3)],'descend');
				dwi_mean_masked = permute(dwi_mean_masked,index_min_mask);
				j_displayMRI(dwi_mean_masked);
				reply1 = input('Do you like this mask? y/n [y]: ', 's');
				if strcmp(reply1,'n')
					txt = ['What FWHM would you like? [previous value was ',num2str(dmri.mask.auto.fwhm),']: '];
					reply2 = input(txt);
					dmri.mask.auto.fwhm = reply2;
					txt = ['What intensity threshold would you like? [previous value was ',num2str(dmri.mask.auto.threshold),']: '];
					reply3 = input(txt);
					dmri.mask.auto.threshold = reply3;
				end
			else
				reply1 = 'y';
			end
			close % close figure
		end

		% Delete datasub
		j_progress('Delete temporary files ........................')
		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},'tmp*.*']);
		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_dwi]);
		j_progress(1)

		% Copy mask in each other folder
		j_progress('Copy mask in each folder ......................')
		for i_folder_sub = 2:nb_folders
			copyfile([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_mask,ext],[dmri.nifti.path,dmri.nifti.folder{i_folder_sub},dmri.nifti.file_mask,ext]);
			j_progress(i_folder_sub/nb_folders)
		end
		if nb_folders==1, j_progress(1); end

	case 'manual'

		disp('-> Mask generation method: "manual"')
		% Ask the user to create a mask...
		disp(['** Open a Terminal and go to the following directory: "',dmri.nifti.path,dmri.nifti.folder{i_folder},'"'])
		disp(['** Then, generate a mask using fslview based on the mean dwi image. To do this, type: "fslview dwi_mean"'])
		disp(['** Once you''re happy with the mask, save it under the name "nodif_brain_mask.nii"'])
		disp(['** Then go back to Matlab and press a key'])
		pause
	end
end % i_folder_mask

% save structure
save([dmri.struct.path,dmri.struct.file],'dmri');










% ====================================================================
% AVERAGE THE DATA 
% ====================================================================
% Average multiple folders
% N.B. Do that direction-by-direction to avoid 'Out of memory'

fprintf('')
j_cprintf('blue','\nAVERAGE THE DATA\n\n')

if dmri.average
	
	% split data by diffusion direction, for each folder
	j_progress('Split data in each folder .....................')
	for i_folder = 1:nb_folders
		fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
		fname_datasub = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub];
		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslsplit ',fname_data,' ',fname_datasub];
		unix(cmd);
		j_progress(i_folder/nb_folders)
	end
	
	% Build folder name
	j_progress('Build folder name .............................')
	dmri.nifti.folder_average = ['average_'];
	for iFolder=1:nb_folders
		dmri.nifti.folder_average	= strcat(dmri.nifti.folder_average,dmri.nifti.folder{iFolder}(1:end-1),'-'); % average multiple folders
	end
	dmri.nifti.folder_average = strcat(dmri.nifti.folder_average(1:end-1),'/');
	j_progress(1)
	
	% create folder (if does not already exist)
	if ~exist(strcat(dmri.nifti.path,dmri.nifti.folder_average))
		j_progress('Create folder .................................')
		mkdir(strcat(dmri.nifti.path,dmri.nifti.folder_average));
		j_progress(1)
	end
	
	% load mask
	j_progress('Load mask (to get the dimensions) .............')
	mask = j_mri_read([dmri.nifti.path,dmri.nifti.folder{1},dmri.nifti.file_mask]);
	dmri.nifti.nx = size(mask,1);
	dmri.nifti.ny = size(mask,2);
	dmri.nifti.nz = size(mask,3);
	j_progress(1)
	
	% loop across diffusion directions
	nx = double(dmri.nifti.nx);
	ny = double(dmri.nifti.ny);
	nz = double(dmri.nifti.nz);
	nxyz = nx*ny*nz;
	
	bvals = j_readFile([dmri.nifti.path,dmri.nifti.folder{1},dmri.nifti.file_bvals]);
	nt = length(bvals);%dmri.nifti.nb_dir+1;
	data2d = zeros(nxyz,nb_folders);
% 	data2d_masked = zeros(nxyz,nb_folders);
% 	data2d_masked_mean = zeros(nxyz,1);
% 	data2d_masked = zeros(length(index_mask),nb_folders);
% 	data2d_masked_mean = zeros(length(index_mask),1);
	data2d_mean = zeros(nxyz,1);
	data_mean = zeros(nx,nx,nz);
	num = j_numbering(nt,4,0);
	j_progress('Average across diffusion directions ...........')
	for it=1:nt
		% loop across folders
		for i_folder = 1:nb_folders
			opt.output = 'nifti';
			nifti = j_mri_read([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub,num{it},'.nii'],opt);
			data2d(:,i_folder) = double(reshape(nifti.img,nxyz,1));
% 			data2d_masked(:,i_folder) = data2d;
% 			data2d_masked(:,i_folder) = data2d(index_mask,:);
		end
		% average data
		data2d_mean = mean(data2d,2);
		clear data2d
		% reshape data
% 		data2d_mean = data2d_mean;
% 		data2d_mean(index_mask,:) = data2d_masked_mean;
		data_mean = reshape(data2d_mean,nx,ny,nz);
		% write data
		nifti.img = data_mean;
		j_save_untouch_nii(nifti,[strcat(dmri.nifti.path,dmri.nifti.folder_average),dmri.nifti.file_datasub,num{it},'.nii']);
		% display progress
		j_progress(it/nt)
	end
	
	% Merge data into 4D file
	j_progress('Merge data into 4D file .......................')
	fname_data_moco_3d = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_datasub,'*.*'];
	fname_data_moco_4d = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_data];
	cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_data_moco_4d,' ',fname_data_moco_3d];
	[status result] = unix(cmd);
	j_progress(1)

	% delete temporary data
	j_progress('Delete temporary files ........................')
	for i_folder = 1:nb_folders
		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub,'*.*']);
	end
	delete([dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_datasub,'*.*']);
	j_progress(1)
	
	% Copy b-matrix file
	j_progress('Copy b-matrix files ...........................')
	copyfile([dmri.nifti.path,dmri.nifti.folder{1},dmri.nifti.file_bvecs],[dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_bvecs]);
	copyfile([dmri.nifti.path,dmri.nifti.folder{1},dmri.nifti.file_bvals],[dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_bvals]);
	j_progress(1)

	% Copy mask
	j_progress('Copy mask .....................................')
	copyfile([dmri.nifti.path,dmri.nifti.folder{1},dmri.nifti.file_mask],[dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_mask]);
	j_progress(1)

else
	
	fprintf(1,'Skip this step.\n');

end










% ====================================================================
% EDDY-CURRENT CORRECTION
% ====================================================================
% Correct for Eddy-currents distortions
% N.B. Do that AFTER averaging to get better estimate of transformation
% matrix for each DW image - which are quite low SNR.

fprintf('')
j_cprintf('blue','\nEDDY-CURRENT CORRECTION\n\n')

if dmri.eddy_correct.do
	
	nt = dmri.nifti.nb_dir;
	j_progress('Remove the B0 image from DWI data .............')
	fname_data = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_data];
	fname_dwi = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi];
	cmd = ['export FSLOUTPUTTYPE=NIFTI; fslroi ',fname_data,' ',fname_dwi,'	1 ',num2str(nt)];
	unix(cmd);
	j_progress(1)

	j_progress('Mean the DW data ..............................')
	fname_dwi = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi];
	fname_dwi_mean{i_folder} = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_mean];
	cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmaths ',fname_dwi,' -Tmean ',fname_dwi_mean{i_folder}];
	unix(cmd);
	j_progress(1)
	
	j_progress('Add the mean DW image to the DW data ..........')
	fname_dwi_mean{i_folder} = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_mean];
	fname_dwi = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi];
	fname_dwi_with_dwi_mean = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_with_dwi_mean];
	cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_dwi_with_dwi_mean,' ',fname_dwi_mean{i_folder},' ',fname_dwi];
	unix(cmd);
	j_progress(1)
	
	j_progress('Correct for Eddy-current distortions ..........')
	fname_dwi_with_dwi_mean = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_with_dwi_mean];
	fname_dwi_with_dwi_mean_eddy = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_with_dwi_mean_eddy];
	cmd = ['export FSLOUTPUTTYPE=NIFTI; eddy_correct ',fname_dwi_with_dwi_mean,' ',fname_dwi_with_dwi_mean_eddy,' 0'];
	unix(cmd);
	j_progress(1)
	
	j_progress('Remove the mean DW image ......................')
	fname_dwi_with_dwi_mean_eddy = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_with_dwi_mean_eddy];
	fname_dwi_eddy = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_eddy];
	cmd = ['export FSLOUTPUTTYPE=NIFTI; fslroi ',fname_dwi_with_dwi_mean_eddy,' ',fname_dwi_eddy,'	1 ',num2str(nt)];
	unix(cmd);
	j_progress(1)

	j_progress('Get the B0 ....................................')
	fname_data = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_data];
	fname_b0 = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_b0];
	cmd = ['export FSLOUTPUTTYPE=NIFTI; fslroi ',fname_data,' ',fname_b0,'	0 1'];
	unix(cmd);
	j_progress(1)

	j_progress('Add the B0 to the DW data .....................')
	fname_dwi_eddy = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_eddy];
	fname_b0 = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_b0];
	fname_data_eddy = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_data_eddy];
	cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_data_eddy,' ',fname_b0,' ',fname_dwi_eddy];
	unix(cmd);
	j_progress(1)

	j_progress('Delete temporary files ........................')
	fname_dwi = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi];
	fname_dwi_mean{i_folder} = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_mean];
	fname_dwi_with_dwi_mean = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_with_dwi_mean];
	fname_dwi_with_dwi_mean_eddy = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_with_dwi_mean_eddy];
	fname_dwi_eddy = [dmri.nifti.path,dmri.nifti.folder_average,dmri.nifti.file_dwi_eddy];
	delete(fname_dwi);
% 	delete(fname_dwi_mean{i_folder});
	delete(fname_dwi_with_dwi_mean);
	delete(fname_dwi_with_dwi_mean_eddy);
	delete(fname_dwi_eddy);

	% update structure: data to use
	dmri.nifti.file_data = dmri.nifti.file_data_eddy;

else
	fprintf(1,'Skip this step.\n');
end













% ====================================================================
% PROCESS DATA
% ====================================================================

fprintf('')
j_cprintf('blue','\nPROCESS THE DATA\n\n')

% Estimate the tensors
if dmri.dti.do_each_run
	j_progress('Estimate DTI for each run .....................')
	for i_folder = 1:nb_folders
		% Go to the data folder
		cd([dmri.nifti.path,dmri.nifti.folder{i_folder}]);
		% estimate tensors using FSL
		cmd = ['dtifit -k ',dmri.nifti.file_data,...
			' -m ',dmri.nifti.file_mask,...
			' -o ',dmri.nifti.file_dti,...
			' -r ',dmri.nifti.file_bvecs,...
			' -b ',dmri.nifti.file_bvals];
		[status result] = unix(cmd);
		j_progress(i_folder/nb_folders)
	end
end

% estimate the tensors for the averaged dataset
j_progress('Estimate DTI for the average run ..............')
if dmri.average
	% Go to the data folder
	cd([dmri.nifti.path,dmri.nifti.folder_average]);
	% estimate tensors using FSL
	cmd = ['dtifit -k ',dmri.nifti.file_data,...
		' -m ',dmri.nifti.file_mask,...
		' -o ',dmri.nifti.file_dti,...
		' -r ',dmri.nifti.file_bvecs,...
		' -b ',dmri.nifti.file_bvals];
	[status result] = unix(cmd);
end
j_progress(1)

% q-ball estimation
if dmri.dtk.do
	j_progress('Estimate Q-Ball ...............................')
	for i_folder = 1:nb_folders
		% get numbers of B0
		% N.B. B0 SHOULD ALWAYS BE AT THE BEGGINING OF THE ACQUISITION!!! (NOT IN THE MIDDLE)
		bvals = textread(fname_bvals);
		nb_b0 = max(find(bvals==0));
		% create a gradient vector list compatible with DTK (should contain no b0)
		fname_bvecs = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvecs];
		bvecs_dtk = textread(fname_bvecs);
		bvecs_dtk_nob0 = bvecs_dtk(nb_b0+1:end,:);
		fname_bvecs_dtk = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.dtk.file_bvecs_dtk];
		fid = fopen(fname_bvecs_dtk,'w');
		fprintf(fid,'%f %f %f \n',bvecs_dtk_nob0);
		fclose(fid);
		% copy matrices file from DTK
		copyfile([dmri.dtk.folder_mat,'*.dat'],'.');
		% estimate q-ball using DTK
		fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
		fname_qball = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.dtk.file_qball];
		nb_dirs = size(bvecs_dtk_nob0,1);
	% 	fprintf('\nEstimate q-ball for nav = %i\n',i_folder)
		cmd = ['hardi_mat ',fname_bvecs_dtk,' ','temp_mat.dat',' -ref ',fname_data];
		[status result] = unix(cmd);
		cmd = ['odf_recon ',fname_data,' ',num2str(nb_dirs+1),' 181 ',fname_qball,' -b0 ',num2str(nb_b0),' -mat temp_mat.dat -nt -p 3 -sn 1 -ot nii'];
		[status result] = unix(cmd);
		% delete temp file
		delete('temp_mat.dat');
		delete('DSI_*.dat')
		j_progress(i_folder/nb_folders)
	end
end

% Come back to the structure folder
cd(dmri.struct.path);


% % Generate shell scripts to launch BEDPOSTX on super-computer
% j_progress('Generate batch to run BedpostX ..........................')
% fname_batch = [dmri.path,filesep,'batch_bedpostx.sh'];
% fid = fopen(fname_batch,'w');
% for i_nex = 1:dmri.nex
% 	fprintf(fid,'echo ******************************************\n');
% 	fprintf(fid,'echo * Process series %s ...\n',['average_01-',num{i_nex}]);
% 	fprintf(fid,'echo ******************************************\n');
% 	fprintf(fid,'bedpostx %s -n 1\n',['average_01-',num{i_nex}]);
% 	j_progress(i_nex/dmri.nex)
% end
% fclose(fid);
% j_progress(1)

% Delete temp files
% j_progress('Delete temp files .............................')
% delete(fname_nodif{i_folder});
% delete(fname_mask);
% j_progress(1)

% COPY B0 and B0_moco!!!

% save structure
j_progress('Save structure ................................')
save([dmri.struct.path,dmri.struct.file],'dmri');
j_progress(1)

disp('** To compute angular uncertainty, go to each folder and type: "bedpostx . -n 2 -j 1000 -s 10"')
disp('** Then go back to Matlab and run "j_dmri_compute_uncertainty" in each folder individually')









% =========================================================================
% FUNCTION
% retrieve_gradients
%
% Retrieve gradient vectors for each header.
% =========================================================================
function [gradient_list] = retrieve_gradient(bvecs)

max_distance_vector = 0.001; % euclidean distance between two gradient vectors considered as being the same
% if ~exist('max_distance_vector'), max_distance_vector = 0.001; end
% 
nb_headers = size(bvecs,1);
nb_directions = 0; % !!! INCLUDES B0 VALUE!!!!
for i_file=1:nb_headers
    % retrieve actual gradient
    gradient_tmp = bvecs(i_file,:);
    % compare actual gradient with previous ones
    found_existing_gradient = 0;
    for i_gradient=1:nb_directions
        distance_vector = (gradient_tmp(1)-gradient_list(i_gradient).direction(1))^2+(gradient_tmp(2)-gradient_list(i_gradient).direction(2))^2+(gradient_tmp(3)-gradient_list(i_gradient).direction(3))^2;
        distance_vector_neg = (gradient_tmp(1)+gradient_list(i_gradient).direction(1))^2+(gradient_tmp(2)+gradient_list(i_gradient).direction(2))^2+(gradient_tmp(3)+gradient_list(i_gradient).direction(3))^2;
        if (distance_vector < max_distance_vector) | (distance_vector_neg < max_distance_vector)
            % attibute present file index to existing direction
            gradient_list(i_gradient).index = cat(1,gradient_list(i_gradient).index,i_file);
            found_existing_gradient = 1;
        end
    end
    if ~found_existing_gradient
        % create new entry
        nb_directions = nb_directions + 1;
        gradient_list(nb_directions).direction = gradient_tmp;
        gradient_list(nb_directions).index = i_file;
    end
end





% OLD CODE
% 				j_cprintf('black','\n')
% 				j_cprintf('-black','RUN #%i',i_folder)
% 				j_cprintf('black','\n')
% 
% 				% find where are the b0
% 				index_b0 = find(dmri.gradients.bvals==0);
% 				nb_subs = length(index_b0);
% 
% 				% split the 4D dataset into 3D
% 				j_progress('Split the 4D dataset into 3D ..................')
% 				fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
% 				fname_datasub = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub];
% 				cmd = ['export FSLOUTPUTTYPE=NIFTI; fslsplit ',fname_data,' ',fname_datasub];
% 				unix(cmd);
% 				j_progress(1)
% 
% 				% find which files are the b0
% 				j_progress('Find which files are the b0 ...................')
% 				nb_files = length(dmri.gradients.bvals);
% 				num = j_numbering(nb_files,4,0);
% 				for i_sub = 1:nb_subs
% 					fname_datasub_b0{i_folder,i_sub} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub,num{index_b0(i_sub)}];
% 				end
% 				j_progress(1)
% 
% 				% estimate motion for each b0
% 				j_progress('Estimate motion for each b0 ...................')
% 				fname_moco_intra = 'temp';
% 				fname_target = fname_datasub_b0{1}; % target is the first b0
% 				fname_mat_intra = [];
% 				for i_sub = 2:nb_subs
% 					fname_mat_intra{i_sub} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_moco_intra_mat,num2str(i_sub)];
% 					cmd = ['flirt -in ',fname_datasub_b0{i_folder,i_sub},' -ref ',fname_target,' -out ',fname_moco_intra,' -omat ',fname_mat_intra{i_sub},' -cost ',dmri.moco_intra.cost];
% 					[status result] = unix(cmd);
% 					j_progress(i_sub/nb_subs)
% 				end
% 
% 				% generate the identity matrix for the first chunck of data (to get the same re-interpolation procedure)
% 				fname_mat_intra{1} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_moco_intra_mat,num2str(1)];
% 				fid = fopen(fname_mat_intra{1},'w');
% 				fprintf(fid,'%i %i %i %i\n',[1 0 0 0]);
% 				fprintf(fid,'%i %i %i %i\n',[0 1 0 0]);
% 				fprintf(fid,'%i %i %i %i\n',[0 0 1 0]);
% 				fprintf(fid,'%i %i %i %i\n',[0 0 0 1]);
% 				fclose(fid);
% 
% 				% If there is inter-run motion corection
% 				if dmri.moco_inter.do
% 					j_progress('Update matrices with inter-run motion .........')
% 					for i_sub = 1:nb_subs
% 						% load intra-reg matrix
% 						mat_intra = textread(fname_mat_intra{i_sub});
% 						R_intra = mat_intra(1:3,1:3);
% 						T_intra = mat_intra(1:3,4);
% 						% load inter-reg matrix
% 						mat_inter = textread(fname_mat_inter{i_folder});
% 						R_inter = mat_inter(1:3,1:3);
% 						T_inter = mat_inter(1:3,4);
% 						% multiply rotation matrices
% 						R_total = R_intra*R_inter;
% 						% add translation matrices
% 						T_total = T_inter+T_intra;
% 						% write new matrix (overwrite the old one)				
% 						mat_total = cat(2,R_total,T_total);
% 						mat_total(4,:) = [0 0 0 1];
% 						fid = fopen(fname_mat_intra{i_sub},'w');
% 						fprintf(fid,'%f %f %f %f\n',mat_total(1,:));
% 						fprintf(fid,'%f %f %f %f\n',mat_total(2,:));
% 						fprintf(fid,'%f %f %f %f\n',mat_total(3,:));
% 						fprintf(fid,'%f %f %f %f\n',mat_total(4,:));
% 						fclose(fid);
% 						j_progress(i_sub/nb_subs)
% 					end
% 				end
% 
% 				% If there is inter-session registration
% 				if dmri.moco_session.do
% 					j_progress('Update matricies with reference scan ..........')
% 					for i_sub = 1:nb_subs
% 						% load intra-reg matrix
% 						mat_intra = textread(fname_mat_intra{i_sub});
% 						R_intra = mat_intra(1:3,1:3);
% 						T_intra = mat_intra(1:3,4);
% 						% load inter-reg matrix
% 						mat_session = textread(fname_mat_session);
% 						R_session = mat_session(1:3,1:3);
% 						T_session = mat_session(1:3,4);
% 						% multiply rotation matrices
% 						R_total = R_intra*R_session;
% 						% add translation matrices
% 						T_total = T_intra+T_session;
% 						% write new matrix (overwrite the old one)				
% 						mat_total = cat(2,R_total,T_total);
% 						mat_total(4,:) = [0 0 0 1];
% 						fid = fopen(fname_mat_intra{i_sub},'w');
% 						fprintf(fid,'%f %f %f %f\n',mat_total(1,:));
% 						fprintf(fid,'%f %f %f %f\n',mat_total(2,:));
% 						fprintf(fid,'%f %f %f %f\n',mat_total(3,:));
% 						fprintf(fid,'%f %f %f %f\n',mat_total(4,:));
% 						fclose(fid);
% 						j_progress(i_sub/nb_subs)
% 					end
% 				end



% case '2d'
% 	for i_folder=1:nb_folders
% 		
%  		j_cprintf('black','\n')
% 		j_cprintf('-black','RUN #%i',i_folder)
% 		j_cprintf('black','\n')
% 
% 		% Get the first #n time series
% 		j_progress('Extract the first #n time series ..............')
% 		fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
% 		fname_data_firstvols = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_firstvols];
% 		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslroi ',fname_data,' ',fname_data_firstvols,' 0 ',num2str(dmri.moco_intra.nbFirstvols)];
% 		[status result] = unix(cmd);
% 		j_progress(1)
% 
% 		% Average the first #n time series
% 		j_progress('Average the first #n time series ..............')
% 		fname_data_firstvols = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_firstvols];
% 		fname_data_firstvols_mean = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_firstvols_mean];
% 		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmaths ',fname_data_firstvols,' -Tmean ',fname_data_firstvols_mean];
% 		[status result] = unix(cmd);
% 		j_progress(1)
% 
% 		% split data in the Z-dimension
% 		j_progress('Split data in the Z-dimension .................')
% 		fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
% 		fname_datasub = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub];
% 		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslsplit ',fname_data,' ',fname_datasub,' -z'];
% 		unix(cmd);
% 		j_progress(1)
% 
% 		% split the ref image in the Z-dimension
% 		j_progress('Split the ref image in the Z-dimension ........')
% 		fname_data_firstvols_mean = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_firstvols_mean];
% 		fname_datasub_ref = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub_ref];
% 		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslsplit ',fname_data_firstvols_mean,' ',fname_datasub_ref,' -z'];
% 		unix(cmd);
% 		j_progress(1)
% 
% 		% estimate motion for each slice
% 		j_progress('Estimate motion with mcflirt ..................')
% 		num = j_numbering(nz,4,0);
% 		for iz=1:nz
% 			fname_datasub = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub,num{iz}];
% 			fname_datamoco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco,num{iz}];
% 			fname_datasub_ref = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub_ref,num{iz}];
% 			cmd = ['export FSLOUTPUTTYPE=NIFTI; mcflirt -in ',fname_datasub,' -out ',fname_datamoco,' -r ',fname_datasub_ref];
% 			[status result] = unix(cmd);
% 			j_progress(iz/nz)
% 		end
% 		
% 		% merge data
% 		j_progress('Merge data into 4D file .......................')
% 		fname_data_moco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_moco_intra];
% 		fname_data_moco_3d = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco,'*.nii'];
% 		cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -z ',fname_data_moco,' ',fname_data_moco_3d];
% 		[status result] = unix(cmd);
% 		j_progress(1)
% 
% 		% Delete temp files
% 		j_progress('Delete temporary files ........................')
% 		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub,'*.*']);
% 		j_progress(0.25)
% 		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco,'*.*']);
% 		j_progress(0.5)
% 		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datasub_ref,'*.*']);
% 		j_progress(0.75)
% 		delete([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_firstvols,'*.*']);
% 		j_progress(1)
% 	end





% 			if dmri.moco_intra.do
% 				disp('-> Motion correction was NOT applied')
% 			end

% 			% Check for multiple nex
% 			if dmri.multiple_nex
% 
% 				% get gradient list
% 				gradient_list = retrieve_gradient(dmri.gradients.bvecs);
% 				nb_dir_new = size(gradient_list,2);
% 				disp(['-> found ',num2str(nb_dir_new-1),'+1 directions'])
% 				bvecs_new = zeros(nb_dir_new,3);
% 				bvals_new = zeros(nb_dir_new,1);
% 				num_dir = j_numbering(nb_dir_new,4,0);
% 
% 				% Merge merge multiple nex
% 				j_progress('Merge multiple nex ............................')
% 				for i_dir = 1:nb_dir_new
% 
% 					% get gradient direction
% 					bvecs_new(i_dir,:) = gradient_list(i_dir).direction;
% 					bvals_new(i_dir) = dmri.gradients.bvals(i_dir);
% 
% 					% get file names
% 					fname_datamoco_merge = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco_merge,'_',num_dir{i_dir}];
% 					cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_datamoco_merge];
% 					nb_nex = size(gradient_list(i_dir).index,1);
% 					for i_nex = 1:nb_nex
% 						% add moco image corresponding to selected diffusion direction and nex
% 						cmd = strcat(cmd,[' ',fname_moco{gradient_list(i_dir).index(i_nex)}]);
% 					end
% 					[status result] = unix(cmd);
% 					j_progress(i_dir/nb_dir_new)
% 				end
% 				disp(['-> ',num2str(nb_nex),' nex'])
% 				disp(['-> ',num2str(size(gradient_list(1).index,1)/nb_nex),' b0/nex'])
% 
% 				% Average multiple nex
% 				j_progress('Average multiple nex ..........................')
% 				for i_dir = 1:nb_dir_new
% 
% 					% get file names
% 					fname_datamoco_merge = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco_merge,'_',num_dir{i_dir}];
% 					fname_datamoco_mean = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco_mean,'_',num_dir{i_dir}];
% 					% average between nex
% 					cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmaths ',fname_datamoco_merge,' -Tmean ',fname_datamoco_mean];
% 					[status result] = unix(cmd);
% 					j_progress(i_dir/nb_dir_new)
% 				end
% 
% 				% Merge b0 images
% 				j_progress('Merge b0 images ...............................')
% 				fname_nodif{i_folder} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra];
% 				fname_nodif_moco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra_moco];
% 				cmd_raw = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_nodif{i_folder}];
% 				cmd_moco = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_nodif_moco];
% 				for i_sub = 1:nb_subs
% 					% raw b0 images
% 					cmd_raw = strcat(cmd_raw,[' ',fname_datasub_b0{i_folder,i_sub},'.*']);
% 					% moco b0 images
% 					fname_b0_moco{i_sub} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco,num{index_b0(i_sub)},'.*'];
% 					cmd_moco = strcat(cmd_moco,[' ',fname_b0_moco{i_sub}]);
% 				end
% 				[status result] = unix(cmd_raw);
% 				[status result] = unix(cmd_moco);
% 				j_progress(1)
% 
% 				% merge data
% 				j_progress('Merge data into 4D file .......................')
% 				fname_data_moco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_moco_intra];
% 				fname_data_moco_3d = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco_mean,'*.nii'];
% 				cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_data_moco,' ',fname_data_moco_3d];
% 				[status result] = unix(cmd);
% 				j_progress(1)
% 
% 				% create new bvecs/bvals files
% 				j_progress('Create new bvecs/bvals files ..................')
% 				fid_bvecs = fopen([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvecs_moco_intra],'w');
% 				fid_bvals = fopen([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvals_moco_intra],'w');
% 				for i_dir=1:size(bvecs_new,1)
% 					fprintf(fid_bvecs,'%f %f %f\n',bvecs_new(i_dir,:));
% 					fprintf(fid_bvals,'%i\n',bvals_new(i_dir));
% 				end
% 				fclose(fid_bvecs);
% 				fclose(fid_bvals);
% 				j_progress(1)
% 
% 			else

% 			% Merge b0 images
% 			j_progress('Merge b0 images ...............................')
% 			fname_nodif{i_folder} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra];
% 			fname_nodif_moco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra_moco];
% 			cmd_raw = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_nodif{i_folder}];
% 			cmd_moco = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_nodif_moco];
% 			for i_sub = 1:nb_subs
% 				% raw b0 images
% 				cmd_raw = strcat(cmd_raw,[' ',fname_datasub_b0{i_folder,i_sub},'.*']);
% 				% moco b0 images
% 				fname_b0_moco{i_sub} = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco,num{index_b0(i_sub)},'.*'];
% 				cmd_moco = strcat(cmd_moco,[' ',fname_b0_moco{i_sub}]);
% 			end
% 			[status result] = unix(cmd_raw);
% 			[status result] = unix(cmd_moco);
% 			j_progress(1)
% 
% 			% Average b0
% 			j_progress('Average b0 images .............................')
% 			fname_b0_mean = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_b0_intra_mean];
% 			cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmaths ',fname_nodif_moco,' -Tmean ',fname_b0_mean];
% 			[status result] = unix(cmd);
% 			j_progress(1)
% 
% 			% Delete b0
% 			j_progress('Delete b0 images ..............................')
% 			for i_sub = 1:nb_subs
% 				delete(fname_b0_moco{i_sub});
% 				j_progress(i_sub/nb_subs)
% 			end
% 
% 			% Add mean b0 as the first volume
% 			j_progress('Add mean b0 as the first volume ...............')
% 			copyfile(fname_b0_mean,[fname_moco{1},'.nii']);
% 			j_progress(1)
% 
% 			% merge data
% 			j_progress('Merge data into 4D file .......................')
% 			fname_data_moco = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data_moco_intra];
% 			fname_data_moco_3d = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_datamoco,'*.nii'];
% 			cmd = ['export FSLOUTPUTTYPE=NIFTI; fslmerge -t ',fname_data_moco,' ',fname_data_moco_3d];
% 			[status result] = unix(cmd);
% 			j_progress(1)
% 
% 			% make clean bvals/bvecs files (remove the b0 inside)
% 			j_progress('Correct bvals/bvecs files .....................')
% 			bvecs = textread([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvecs]);
% 			bvals = textread([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvals]);
% 			i_sub = 2;
% 			i_file_new = 1;
% 			bvecs_new = [];
% 			bvals_new = [];
% 			for i_file = 1:nb_files
% 				if i_file~=index_b0(i_sub)
% 					bvecs_new(i_file_new,:) = bvecs(i_file,:);
% 					bvals_new(i_file_new,:) = bvals(i_file,:);
% 					i_file_new = i_file_new+1;
% 				else
% 					i_sub = i_sub+1;
% 				end
% 			end
% 			% TODO: FIND A WAY TO CORRECT B-MATRIX DUE TO SUBJECT MOTION
% 			% create new bvecs/bvals files
% 			fid_bvecs = fopen([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvecs_moco_intra],'w');
% 			fid_bvals = fopen([dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvals_moco_intra],'w');
% 			for i_dir=1:size(bvecs_new,1)
% 				fprintf(fid_bvecs,'%f %f %f\n',bvecs_new(i_dir,:));
% 				fprintf(fid_bvals,'%i\n',bvals_new(i_dir));
% 			end
% 			fclose(fid_bvecs);
% 			fclose(fid_bvals);
% 			j_progress(1)

% 			end % multiple nex

				