
% PUT THAT IN A BATCH FILE AND THEN CALL THIS SCRIPT
% 
% % user parameters
% i_folder		= 1;
% method			= 'qball'; % 'qball' (Descoteaux), 'csd' (constrained spherical deconvolution, Tournier)
% 
% order_odf		= 4;
% sharpening		= 0; % sharpening of the Q-Ball ODF . N.B. Only used for the Descoteaux method. Put 0 for no sharpening.
% lambda			= 0.006; % regularization parameter (default is 0.006)
% 
% sampling_odf	= 362; % sampling of the ODF (only for visualization purpose)
% order_dwi		= 16; % SH order for DWI images (before qball or csd estimation)
% order_csd		= 4; % SH order of the ODF
% order_visu		= 16;
% prefixe_mask	= 'b3000_';
% fname_data		= '/Users/julien/MRI/connectome/31/21/21/data_moco_intra_cortex.nii.gz';%'/Users/julien/mri/connectome/22/average/data_moco.nii';
% fname_mask		= '/Users/julien/mri/connectome/31/21/21/data_moco_intra_cortex-mask.nii.gz';
% 
% % debug parameters
% load_data		= 1;
% nx_odf			= 0;
% ny_odf			= 0;

% default parameters
sh_order = [0 6 0 15 0 28 0 45 0 0 0 91 0 0 0 153]; % nb of coefficients given the max order of SH decomposition


load dmri

nb_b0 = dmri.nifti.nb_b0;
nb_dir = dmri.nifti.nb_dir;

fprintf('\n')

if load_data
	% load data
	j_progress('Load data .................')
	% fname_data = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_data];
	nifti = j_mri_read(fname_data);
	data = nifti.img;
	clear nifti
	j_progress(0.5)
	% add Z dimension if necessary
	if length(size(data))==3
		data = permute(data,[1 2 4 3]);
	end
	nx = size(data,1);
	ny = size(data,2);
	nz = size(data,3);

	% remove b0
	if ~strcmp(method,'dti')
		data = data(:,:,:,nb_b0+1:end);
	end
	
	% load mask
	nifti = j_mri_read(fname_mask);
	mask = nifti.img;
	clear nifti
	% add Z dimension if necessary
% 	if length(size(mask))==2
% 		mask = permute(mask,[1 2 3]);
% 	end
	
	% load bvecs
% 	fname_bvecs = [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvecs];
	% [dmri.nifti.path,dmri.nifti.folder{i_folder},dmri.nifti.file_bvecs];
	fid = fopen(fname_bvecs,'r');
	bvecs = textscan(fid,'%f %f %f');
	bvecs = cell2mat(bvecs);
	if ~strcmp(method,'dti')
		bvecs = bvecs(nb_b0+1:end,:);
	else
		nb_dir=nb_dir+1;
	end
	fclose(fid);

	% reshape data
	mask2d = reshape(mask,1,nx*ny*nz);
	index_mask = find(mask2d);
	data2d = double(reshape(data,nx*ny*nz,nb_dir));
	data2d_masked = data2d(index_mask,:);

	j_progress(1)
end

% Find the size of the mask
z=[];
for i=1:nz
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
nx_mask = index_xmax-index_xmin+1;
ny_mask = index_ymax-index_ymin+1;
nz_mask = index_zmax-index_zmin+1;
% find slice orientation (according to MNI template)
if nx_mask==1
	orientation = 'axial';
	nx_mask = nz_mask;
elseif ny_mask==1
	orientation = 'coronal';
	ny_mask = nz_mask;
elseif nz_mask==1
	orientation = 'sagittal';
end

% flip bvecs
bvecs_temp = bvecs;
for i=1:nb_dir
	switch (orientation)
		
	case 'coronal'
		bvecs_temp(i,1) = bvecs(i,1);
		bvecs_temp(i,2) = bvecs(i,3);
		bvecs_temp(i,3) = bvecs(i,2);
% 		bvecs_temp(i,1) = bvecs(i,1);
% 		bvecs_temp(i,2) = bvecs(i,3);
% 		bvecs_temp(i,3) = bvecs(i,2);
	end
end
bvecs = bvecs_temp;

% estimate ODF
nb_voxels = size(data2d_masked,1);
j_progress('Estimate Q-Ball ...........')
switch(method)

% estimate tensor
case 'dti'
	fa2d = zeros(nb_voxels,1);
	tensor2d = zeros(nb_voxels,6);
	% Define samplings for inputOrientationSet for order
	scheme_dw = gen_scheme(bvecs, order_qball);
	% Convert to cartesian coords
	cart=scheme_dw.vert;
	% Determine direction matrix X
	X = zeros(nb_dir-1,6);
	for j=1:nb_dir-1
		X(j,:) = [cart(j,1).^2 2*cart(j,1)*cart(j,2) 2*cart(j,1)*cart(j,3) cart(j,2).^2 2*cart(j,2)*cart(j,3) cart(j,3).^2];
	end
	% loop over voxels
	for i=1:nb_voxels

		S = data2d_masked(i,2:end)/data2d_masked(i,1);	
		% compute tensor
	    tensor2d(i,:) = inv(X'*X)*X'* (-log(S')/bvalue);
% 		% compute eigenvalues
% 		eig = svd([D(1) D(2) D(3);D(2) D(4) D(5);D(3) D(5) D(6)]);
% 		% project data on a sampled sphere (for visualization)
% 		tensor2d(i,:) = [scheme_visu.vert(:,1).* eig(1) scheme_visu.vert(:,2).* eig(2) scheme_visu.vert(:,3).* eig(3)];
% 		j_progress(i/nb_voxels)
	end

	% reshape
% 	fa = reshape(fa2d,nx_mask,ny_mask);
	tensor = reshape(tensor2d,nx_mask,ny_mask,6);
	

% estimate q-ball (Max method)
case 'qball'
	gfa2d = zeros(nb_voxels,1);
	qball2d = zeros(nb_voxels,sh_order(order_qball));
	% Define samplings for inputOrientationSet for order
	scheme_dw = gen_scheme(bvecs, order_qball);
	for i=1:nb_voxels
		S = data2d_masked(i,:);
		S_SH = amp2SH(S',scheme_dw);
		[qball_tmp gfa_tmp] = estimate_qball(S_SH,order_qball,lambda,sharpening);
		gfa2d(i,1)=gfa_tmp;
		qball2d(i,:) = qball_tmp;
		j_progress(i/nb_voxels)
	end

	% reshape
	gfa = reshape(gfa2d,nx_mask,ny_mask);
	figure, imagesc(fliplr(flipud(gfa')))
	axis image
	colormap gray
	colorbar

	% reshape
	odf = reshape(qball2d,nx_mask,ny_mask,sh_order(order_qball));
	close
	% figure, plot_amp(odf,scheme_odf)


% estimate q-ball (Tournier method)
case 'csd'
	% generate response function with FA, b-value = 3x10^3 s/mm^2
	% this should be estimated from your real data!!!
	% TODO
	% Define samplings for inputOrientationSet for order
	scheme_dw = gen_scheme(bvecs, order_dwi);
	% Define samplings for regularization (constrained spherical deconvolution)
	scheme_hr = gen_scheme ('dir300.txt', order_csd);
	gfa2d = zeros(nb_voxels,1);
	qball2d = zeros(nb_voxels,sh_order(order_csd));
	S_SH_qball = [];
% 	generate_response_function(data2d_masked);
	FA = 0.7;
	b = 1;
	R_SH = amp2SH (eval_DT (FA, b, scheme_dw,0,0), scheme_dw);
	% TODO: CHANGE THE scheme_odf!!! it has nothing to do here!!!
	R_RH = SH2RH (R_SH);
	% loop over voxels
	for i=1:nb_voxels
		S = data2d_masked(i,:);
		S_SH = amp2SH(S',scheme_dw);
        % perform constrained spherical deconvolution (Tournier 2007, Descoteaux TMI 2009)
        [ csd_tmp, num_it ] = csdeconv (R_RH, S_SH, scheme_hr);
		qball2d(i,:) = csd_tmp;
		% estimate qball just to get the GFA
		scheme_dw_qball = gen_scheme(bvecs, order_qball);
		S_SH_qball = amp2SH(S',scheme_dw_qball);
		[qball_tmp gfa_tmp] = estimate_qball(S_SH_qball,order_qball,0.006,sharpening);
		gfa2d(i,1)=gfa_tmp;
		j_progress(i/nb_voxels)
	end
	
	% reshape
	gfa = reshape(gfa2d,nx_mask,ny_mask);
	figure, imagesc(fliplr(flipud(gfa')))
	axis image
	colormap gray
	colorbar

	% reshape
	odf = reshape(qball2d,nx_mask,ny_mask,sh_order(order_csd));
	close
	% figure, plot_amp(odf,scheme_odf)
end

% Define samplings for visu
scheme_visu = gen_scheme(sampling_odf,order_visu);
% display ODF
if ~nx_odf, nx_odf=nx_mask; end
if ~ny_odf, ny_odf=ny_mask; end

if strcmp(method,'dti')
	h_fig = j_display_odf(tensor,'coord','tensor','scheme',scheme_visu,'nx',nx_odf,'ny',ny_odf,'normalize',1,'overlay_gfa',1);
else
	h_fig = j_display_odf(odf,'coord','sh','scheme',scheme_visu,'gfa',gfa,'nx',nx_odf,'ny',ny_odf);
end

% save figure
set(h_fig,'PaperPositionMode','auto')
switch(method)
	case 'dti'
	print(h_fig,'-dpng','-r200',['fig_odf_',prefixe_mask,method,'_lODF',num2str(order_qball),'_sharp',num2str(sharpening),'_lambda',num2str(lambda),'_r300.png']);

	case 'qball'
	print(h_fig,'-dpng','-r200',['fig_odf_',prefixe_mask,method,'_lODF',num2str(order_qball),'_sharp',num2str(sharpening),'_lambda',num2str(lambda),'_r300.png']);

	case 'csd'
	print(h_fig,'-dpng','-r200',['fig_odf_',prefixe_mask,method,'_lDWI',num2str(order_dwi),'_lODF',num2str(order_csd),'_lVISU',num2str(order_visu),'.png']);
end
