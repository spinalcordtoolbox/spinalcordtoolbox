% =========================================================================
% FUNCTION
% dmri_bootstrap_process.m
%
% Process bootstrap data.
% N.B. Run dmri_bootstrap_generate() first!!!
% 
% COMMENTS
% Julien Cohen-Adad 2009-10-30
% =========================================================================
function bootstrap = j_dmri_bootstrap_process_v3(bootstrap)


% Parameters
sh_order = [0 6 0 15 0 28 0 45 0 0 0 91 0 0 0 153]; % nb of coefficients given the max order of SH decomposition


j_cprintf('');
j_cprintf('blue','\nPROCESS BOOTSTRAP DATA\n\n')

% buid file name
num = j_numbering(bootstrap.nb_bootstraps,4,1);
for i_bootstrap = 1:bootstrap.nb_bootstraps
    bootstrap.fname_data{i_bootstrap} = [bootstrap.path,'data_',num{i_bootstrap}];
end

if bootstrap.process_dti
	% process bootstrap data with DTI
	j_progress('Estimate DTI on generated data ................')
	fname_bvecs = [bootstrap.nifti.path,bootstrap.nifti.folder_average,bootstrap.nifti.file_bvecs_moco_intra];
	fname_bvals = [bootstrap.nifti.path,bootstrap.nifti.folder_average,bootstrap.nifti.file_bvals_moco_intra];
	for i_bootstrap = 1:nb_bootstraps
		
		% estimate tensors using FSL
		fname_data = [fname_write{i_bootstrap},'.nii'];
		fname_dti = [bootstrap.path,bootstrap.file_dti,'_',num{i_bootstrap}];
		cmd = ['dtifit -k ',fname_data,...
			' -m ',fname_data,...
			' -o ',fname_dti,...
			' -r ',fname_bvecs,...
			' -b ',fname_bvals];
		[status result] = unix(cmd);

		j_progress(i_bootstrap/nb_bootstraps)
	end
	j_progress(i_bootstrap/nb_bootstraps)
end


% load bvecs
j_progress('Load bvecs file ...............................')
fname_bvecs = [bootstrap.nifti.path,bootstrap.nifti.folder{1},bootstrap.nifti.file_bvecs];
bvecs = textread(fname_bvecs);

% remove b=0
bvecs = bvecs(2:end,:);
j_progress(1)

% flip bvecs for visu
j_progress('Flip bvecs for visualisation purpose ..........')
bvecs_temp = bvecs;
for it=1:size(bvecs,1)
	for iDim=1:size(bootstrap.visu.flip,2)
		bvecs_temp(it,iDim) = (bootstrap.visu.flip(iDim)/abs(bootstrap.visu.flip(iDim)))*bvecs(it,abs(bootstrap.visu.flip(iDim)));
	end
end
bvecs = bvecs_temp;
clear bvecs_temp
j_progress(1)
disp(['Flip is set to: [',num2str(bootstrap.visu.flip(1)),' ',num2str(bootstrap.visu.flip(2)),' ',num2str(bootstrap.visu.flip(3)),']'])

% check null bvecs
j_progress('Check null bvecs ..............................')
iNonNullBvecs = 1;
iNullBvecs = 1;
index_nonNullBvecs = [];
index_NullBvecs = [];
for iBvecs=1:size(bvecs,1)
    if norm(bvecs(iBvecs,:))~=0
        index_nonNullBvecs(iNonNullBvecs) = iBvecs;
        iNonNullBvecs = iNonNullBvecs+1;
    else
        index_NullBvecs(iNullBvecs) = iBvecs;
        iNullBvecs = iNullBvecs+1;
    end
end
j_progress(1)
disp(['Null bvecs found: ',num2str(index_NullBvecs)])

% load mask
j_progress('Load cropped mask .............................')
mask = logical(read_avw([bootstrap.nifti.path,bootstrap.nifti.file_mask_crop]));
mask = reshape(mask,bootstrap.nx,bootstrap.ny);
mask_new = reorient(mask,bootstrap.orientation);
j_progress(1)
disp(['-> Cropped mask: "',bootstrap.nifti.file_mask_crop,'"'])

% Find zeros-values on the data (to speed-up things)
j_progress('Find zeros-values on the data .................')
data3d = read_avw([bootstrap.fname_data{1},'.nii']);
icount=1;
for ix=1:bootstrap.nx
	for iy=1:bootstrap.ny
		if data3d(ix,iy,1) == 0
			mask(ix,iy) = 0;
		end
		j_progress(icount/(bootstrap.nx*bootstrap.ny))
		icount = icount+1;
	end
end
disp(['-> Number of bootstraps: ',num2str(bootstrap.nb_bootstraps)])
disp(['-> Total number of voxels to process: ',num2str(length(find(mask)))])
disp(['-> Method for Q-Ball estimation: "',bootstrap.qball.method,'"'])

% Preallocation
gfa = zeros(bootstrap.nb_bootstraps,bootstrap.nx,bootstrap.ny);
qball = zeros(bootstrap.nb_bootstraps,bootstrap.nx,bootstrap.ny,sh_order(bootstrap.qball.order_odf));
gfa_new = zeros(bootstrap.nb_bootstraps,size(mask_new,1),size(mask_new,2));
qball_new = zeros(bootstrap.nb_bootstraps,size(mask_new,1),size(mask_new,2),sh_order(bootstrap.qball.order_odf));

% HARD DRIVE preallocation
bootstrap.qball.odf_sh = qball;
bootstrap.qball.gfa = gfa;
save([bootstrap.path,bootstrap.file_struct,bootstrap.file_struct_suffixe],'bootstrap');

% process bootstrap data with Q-Ball
j_progress('Estimate Q-Ball on bootstrap data .............')
switch(bootstrap.qball.method)

case 'dtk'
	
	% copy matrices file from DTK
	copyfile([bootstrap.dtk.folder_mat,'*.dat'],'.');
	fname_bvecs_dtk = [bootstrap.nifti.path,bootstrap.nifti.folder{1},bootstrap.dtk.file_bvecs_dtk];
	for i_bootstrap = 1:nb_bootstraps
		% estimate q-ball using DTK
		fname_qball{i_bootstrap} = [bootstrap.path,bootstrap.file_qball,'_',num{i_bootstrap}];
		cmd = ['hardi_mat ',fname_bvecs_dtk,' ','temp_mat.dat',' -ref ',fname_write{i_bootstrap}];
		[status result] = unix(cmd);
		cmd = ['odf_recon ',fname_write{i_bootstrap},' ',num2str(nt-nb_b0+1),' 181 ',fname_qball{i_bootstrap},' -b0 ',num2str(nb_b0),' -mat temp_mat.dat -nt -p 3 -sn 1 -ot nii'];
		[status result] = unix(cmd);
		j_progress(i_bootstrap/nb_bootstraps)
	end
	% delete temp file
	delete('temp_mat.dat');
	delete('DSI_*.dat')

case 'max'

	% create sheme to be used for SH decomposition
	scheme_odf = j_gen_scheme(bvecs, bootstrap.qball.order_odf);

	for i_bootstrap = 1:bootstrap.nb_bootstraps
		
		% load bootstrap data
		data3d = read_avw([bootstrap.fname_data{i_bootstrap},'.nii']);

		% estimate qball
		for ix=1:bootstrap.nx
			for iy=1:bootstrap.ny
				% is this voxel masked?
				if mask(ix,iy)
					if isempty(find(data3d(ix,iy,:)==0))
						S = double(squeeze(data3d(ix,iy,bootstrap.nifti.nb_b0+1:end)));
						S_SH = j_amp2SH(S(index_nonNullBvecs),scheme_odf);
						[qball(i_bootstrap,ix,iy,:) gfa(i_bootstrap,ix,iy)] = j_estimate_qball(S_SH,bootstrap.qball.order_odf,bootstrap.qball.lambda,bootstrap.qball.sharpening);
					end
				end
			end
		end
		j_progress(i_bootstrap/bootstrap.nb_bootstraps)
	end

case 'csd'

	% generate response function with FA, b-value = 3x10^3 s/mm^2
	% this should be estimated from your real data!!!
	% TODO
	% Define samplings for inputOrientationSet for order
	scheme_dwi = j_gen_scheme(bvecs, bootstrap.qball.order_dwi);
	% Define samplings for regularization (constrained spherical deconvolution)
	scheme_hr = j_gen_scheme ('dir300.txt', bootstrap.qball.order_csd);
	gfa2d = zeros(1,1);
	qball2d = zeros(1,sh_order(bootstrap.qball.order_csd));
	S_SH_qball = [];
% 	generate_response_function(data2d_masked);
	FA = 0.7;
	b = 1;
	R_SH = j_amp2SH (j_eval_DT (FA, b, scheme_dwi,0,0), scheme_dwi);
	% TODO: CHANGE THE scheme_odf!!! it has nothing to do here!!!
	R_RH = j_SH2RH (R_SH);
	
	for i_bootstrap = 1:bootstrap.nb_bootstraps
		
		% load bootstrap data
		data3d = read_avw([bootstrap.fname_data{i_bootstrap},'.nii']);

		% estimate qball
		for ix=1:bootstrap.nx
			for iy=1:bootstrap.ny
% 				for iz=1:bootstrap.nz
					
					S = double(squeeze(data3d(ix,iy,bootstrap.nifti.nb_b0+1:end)));
					S_SH = j_amp2SH(S,scheme_dwi);
					% perform constrained spherical deconvolution (Tournier 2007, Descoteaux TMI 2009)
					[ csd_tmp, num_it ] = j_csdeconv (R_RH, S_SH, scheme_hr);
					qball(i_bootstrap,ix,iy,:) = csd_tmp;
% 				end
			end
		end
		j_progress(i_bootstrap/bootstrap.nb_bootstraps)
	end
end % switch

% reorient data (for subsequent displaying under matlab)
j_progress('Reorient data (for display purpose) ...........')
for ib = 1:bootstrap.nb_bootstraps
	gfa_new(ib,:,:) = reorient(squeeze(gfa(ib,:,:)),bootstrap.orientation);
	for it = 1:sh_order(bootstrap.qball.order_odf)
		qball_new(ib,:,:,it) = reorient(squeeze(qball(ib,:,:,it)),bootstrap.orientation);
	end
end
j_progress(1)

% reorient mask
mask = mask_new;

% update nx and ny
bootstrap.nx = size(mask,1);
bootstrap.ny = size(mask,2);

% clear memory
clear gfa qball

% create sheme for visu
scheme_visu = j_gen_scheme(362, 12);

% save structure
bootstrap.qball.scheme_odf = scheme_odf;
bootstrap.qball.scheme_visu = scheme_visu;
bootstrap.mask = mask;
bootstrap.qball.odf_sh = qball_new;
bootstrap.qball.gfa = gfa_new;
% save([bootstrap.nifti.path,bootstrap.struct_process],'dmri');
save([bootstrap.path,bootstrap.file_struct,bootstrap.file_struct_suffixe],'bootstrap');

% j_display_odf(squeeze(bootstrap.qball.odf_sh(1,:,:,:)),'coord','sh','scheme',scheme_visu,'nx',15,'ny',15,'overlay',squeeze(bootstrap.qball.gfa(1,:,:)));
 



% =========================================================================
% =========================================================================
function img_new = reorient(img,orientation)

switch(orientation)
	
case 'axial'
% 	img_new = flipud(fliplr(img'));
	img_new = flipud((img'));
case 'coronal'
	img_new = flipud(fliplr(img'));

end


% OLD CODE

% 		% estimate DTI using DTK
% 		fname_dti{i_bootstrap} = [dmri.nifti.path,bootstrap.folder,bootstrap.file_dti,'_',num{i_bootstrap}];
% 		cmd = ['dti_recon ',fname_write{i_bootstrap},' ',fname_dti{i_bootstrap},' -gm ',fname_bvecs_dtk,' -b ',num2str(bvalue),' -b0 ',num2str(nb_b0),' -p 3 -sn 1 -ot nii'];
% 		[status result] = unix(cmd);

%		write qball
% 		fname_qball{i_bootstrap} = [dmri.nifti.path,bootstrap.folder,bootstrap.file_qball,'_',bootstrap.num{i_bootstrap}];
% 		hdr_new = hdr;
% 		hdr_new.hdr.dime.dim(5) = sh_order(bootstrap.qball.order_odf);
% 		hdr_new.img = qball;
% 		j_save_untouch_nii(hdr_new,fname_qball{i_bootstrap});
