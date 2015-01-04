% =========================================================================
% FUNCTION
% j_mri_superresolution
%
% Superresolution reconstruction.
%
% INPUTS
% superres				structure
% 	path_exp			= '/Users/julien/MRI/cat/13/02';
% 	num_run				= {{'04','09'},{'05','10'},{'08','11'}}; % epi-,epiShift-,epi+,epiShift+,dwi+,dwiShift+
% 	epi_suffixe			= '-ep2d_diff_cat_DTI'; % folder suffixe name
% 	prefixe_input		= 'cat';
% 	imformat			= 'nii'; % 'nii' or 'img'
% 	prefixe_output		= 'super';
% 	shift				= 1;
%	moco.todo			= 1; % apply motion correctio before reconstruction
%	superres.moco.target	= {'09','10','11'}; % image target for motion correction (put the same number of target as the number of run pairs)
% 
% OUTPUTS
% superres				structure
%
% COMMENTS
% Julien Cohen-Adad 2009-02-20
% =========================================================================
function superres = j_mri_superresolution(superres,i_path,apply_only)


% default initialization
if ~exist('superres'), help j_mri_superresolution; return; end
j_parameters
path_fsl = param.fsl.path;

% create folder
superres.folder{i_path} = strcat('superres_',superres.num_run{i_path}{1},'-',superres.num_run{i_path}{2});
path_superres = strcat(superres.path_exp,filesep,superres.folder{i_path});
if ~exist(path_superres), mkdir(path_superres); end

% retrieve file names
file_name = dir(strcat(superres.path_exp,filesep,superres.num_run{i_path}{1},superres.epi_suffixe,filesep,superres.prefixe_input,'*.',superres.imformat));
file_name_shift = dir(strcat(superres.path_exp,filesep,superres.num_run{i_path}{2},superres.epi_suffixe,filesep,superres.prefixe_input,'*.',superres.imformat));

% display progress
if apply_only, j_progress('Apply transformation .....................'); end

% loop over files
for i_file=1:length(file_name)

	% build full file names
	fname_source = strcat(superres.path_exp,filesep,superres.num_run{i_path}{1},superres.epi_suffixe,filesep,file_name(i_file).name);	
	fname_target = strcat(superres.path_exp,filesep,superres.num_run{i_path}{2},superres.epi_suffixe,filesep,file_name_shift(i_file).name);	

	% check interleaved order
% 	[data hdr_source] = j_cat_mri_read(fname_source);
% 	[data hdr_target] = j_cat_mri_read(fname_target);
% 	diff_mat = sum(hdr_source(1).mat(1:3,4) - hdr_target(1).mat(1:3,4));
	
 	if superres.invertOrder
		shift = [0 0 2]; % used for motion correction
% 		% change the order for interleaving
% 		fname_tmp = fname_target;
% 		fname_target = fname_source;
% 		fname_source = fname_tmp;
% 		clear fname_tmp
		% copy masks in the right folder
		% TODO
	else
		shift = [0 0 -2]; % used for motion correction
 	end
		
	% build transformation file name
	fname_transfo = [superres.path_exp,filesep,superres.file_transfo];

	% estimate AND apply?
	if ~apply_only
		% build mask file name
		if ~isempty(superres.file_mask)
% 			if superres.invertOrder
% 				fname_mask = [fileparts(fname_source),filesep,superres.file_mask];
% 			else
				fname_mask = [fileparts(fname_target),filesep,superres.file_mask];
% 			end
		else
			fname_mask = '';
		end
		% estimate AND apply transformation
		fname_reg = motion_correction(fname_source,fname_target,fname_transfo,fname_mask,apply_only,shift);
	else
		% apply motion correction using the transformation estimated at
		% previous step (used for DWI series)
		fname_reg = motion_correction(fname_source,fname_target,fname_transfo,'',apply_only,shift);
	end

	% build interleaved volume
	if ~apply_only, j_progress('Reconstruct using interleaved method .....'); end
	fname_superres = j_prepend(fname_target,'superres_');
 	if superres.invertOrder
		cmd = [path_fsl,'/bin/fslinterleave ',fname_target,' ',fname_reg,' ',fname_superres];
	else
		cmd = [path_fsl,'/bin/fslinterleave ',fname_reg,' ',fname_target,' ',fname_superres];
	end
	unix(cmd);
	if ~apply_only, j_progress(1); end
	
	% uncompress data
	gunzip([fname_superres,'.gz']);
	delete([fname_superres,'.gz']);

	% move volume to appropriate folder
	movefile(fname_superres,path_superres);

	% delete temporary files
	delete(fname_reg);
	
	% display progress
	if apply_only, j_progress(i_file/length(file_name)); end

end





% =========================================================================
function fname_reg = motion_correction(fname_source,fname_target,fname_transfo,fname_mask,apply_only,shift)

prefixe_write		= 'r';
smooth_window		= [0,0,4];

% warning off % because of the ISFINITE command used in SPM5

% get FSL path
j_parameters
path_fsl = param.fsl.path;

% make file name FSL-friendly
fname_source = fname_source(1:end-4);
fname_target = fname_target(1:end-4);
prefixe = '';

% check if transformation has already been estimated
if ~apply_only

	% smooth images
	j_progress('Smooth data ..............................');
	prefixe = 's';
	fname_in = fname_source;
	cmd = [path_fsl,'/bin/fslmaths ',fname_in,' -kernel gauss 1 -fmean ',j_prepend(fname_in,prefixe)];
	unix(cmd);
	fname_in = fname_target;
	cmd = [path_fsl,'/bin/fslmaths ',fname_in,' -kernel gauss 1 -fmean ',j_prepend(fname_in,prefixe)];
	unix(cmd);
	j_progress(1)

	% translate source data through plane for the subsequent motion correction 
	j_progress('Translate data through-plane .............');
	fid = fopen('transfo_shift','w');
	fprintf(fid,'1 0 0 %i\n',shift(1));
	fprintf(fid,'0 1 0 %i\n',shift(2));
	fprintf(fid,'0 0 1 %i\n',shift(3));
	fprintf(fid,'0 0 0 1');
	fclose(fid);
	fname_in = j_prepend(fname_source,'s');
	prefixe = 't';
	cmd = [path_fsl,'/bin/flirt -in ',fname_in,' -ref ',fname_source,' -out ',j_prepend(fname_in,prefixe),' -init transfo_shift -applyxfm'];
	unix(cmd);
	delete('transfo_shift');
	j_progress(1);

	% find the 2D rigid transformation matrix to register both images
	j_progress('Estimate rigid-body transformation .......');
	fname_in_source = j_prepend(fname_source,'ts');
	fname_in_target = j_prepend(fname_target,'s');
	% use a mask
	if ~isempty(fname_mask)
		% dilate masks
		cmd = [path_fsl,'/bin/fslmaths ',fname_mask,' -kernel boxv 9x9x1 -dilM ',fname_mask,'_dilate'];
		unix(cmd);
		% estimate transformation
		cmd = [path_fsl,'/bin/flirt -in ',fname_in_source,' -ref ',fname_in_target,' -inweight ',fname_mask,'_dilate -omat ',fname_transfo,' -2D -cost normcorr -searchcost normcorr'];
% 		cmd = [path_fsl,'/bin/flirt -in ',fname_in_source,' -ref ',fname_in_target,' -inweight ',fname_mask,'_dilate -omat ',fname_transfo,' -dof 12 -cost corratio'];
	else
		% estimate transformation
		cmd = [path_fsl,'/bin/flirt -in ',fname_in_source,' -ref ',fname_in_target,' -omat ',fname_transfo,' -2D -cost normcorr'];
	end		
	unix(cmd);
	j_progress(1)

end

% apply the estimated rigid-body transformation to the 2nd raw image 
% (i.e. the one used as an input in step 1)
if ~apply_only, j_progress('Apply transformation .....................'); end
fname_reg = j_prepend(fname_source,'r');
cmd = [path_fsl,'/bin/flirt -in ',fname_source,' -ref ',fname_target,' -out ',fname_reg,' -init ',fname_transfo,' -applyxfm'];
unix(cmd);
gunzip([fname_reg,'.nii.gz']);
delete([fname_reg,'.nii.gz']);
fname_reg = [fname_reg,'.nii'];
if ~apply_only, j_progress(1); end

% delete temporary files
if ~apply_only
	j_progress('Delete temporary files ...................');
	delete([j_prepend(fname_source,'s'),'.*']);
	delete([j_prepend(fname_source,'ts'),'.*']);
	delete([j_prepend(fname_target,'s'),'.*']);
	delete([fname_mask,'_dilate.*']);
	j_progress(1);
end




% =========================================================================
% OLD CODE
% =========================================================================

% 		% mask data
% 		j_progress('Mask data ................................');
% 		prefixe = 'm';
% 		fname_in = j_prepend(fname_source,'ts');
% 		cmd = [path_fsl,'/bin/fslmaths ',fname_in,' -mas ',fname_mask,'_dilate ',j_prepend(fname_in,prefixe)];
% 		unix(cmd);
% 		j_progress(0.5)
% 		fname_in = j_prepend(fname_target,'s');
% 		cmd = [path_fsl,'/bin/fslmaths ',fname_in,' -mas ',fname_mask,'_dilate ',j_prepend(fname_in,prefixe)];
% 		unix(cmd);
% 		j_progress(1)



% 		% load volumes
% 		[data hdr] = j_cat_mri_read(fname);
% 		[data_shift hdr_shift] = j_cat_mri_read(fname_shift);
% 		
% 		% evaluate shift through plane
% 		shift = round(hdr_shift.mat(1,4) - hdr.mat(1,4));
% 		
% 		% super-resolution method
% 		switch superres.method
% 			
% 			case 'mean'
% 			% reslice shifted volume
% 			data_shift_resliced = zeros(size(data));
% 			for i_slice = 1:size(data,3);
% 				if ~(i_slice-shift<1 | i_slice-shift>size(data,3))
% 					data_shift_resliced(:,:,i_slice) = data_shift(:,:,i_slice-shift);
% 				end
% 			end
% 			% average both images
% 			data_superres = zeros(size(data));
% 			data_superres = (data + data_shift_resliced) / 2;
% 			
% 			case 'interleave'

% 		
% 			% modify header
% 			data_shift_resliced = zeros(size(data,1),size(data,2),size(data,3)*2);
% 			hdr_superres = hdr;
% 			hdr_superres.dim(3) = hdr_superres.dim(3)*2;
% 			hdr.mat(:,3) = hdr.mat(:,3)/2;
% 			hdr.private.dat.dim(3) = hdr.private.dat.dim(3)*2;
% 			hdr.private.mat(:,3) = hdr.private.mat(:,3)/2;
% 			
% 			% reslice data with interleaved method
% 			for i_slice = 1:size(data,3);
% 				if (i_slice-shift>0)
% 					data_superres(:,:,2*i_slice-1) = data(:,:,i_slice);
% 					data_superres(:,:,2*i_slice) = data_shift(:,:,i_slice);					
% 				else
% 					data_superres(:,:,2*i_slice-1) = data_shift(:,:,i_slice);
% 					data_superres(:,:,2*i_slice) = data(:,:,i_slice);					
% 				end
% 			end
% 		end
% 		
% 		% write volume
% 		j_mri_write(data_superres,hdr_superres,'superres_');
