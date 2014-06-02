function j_dmri_compute_qball(fname_data, fname_bvecs, opt)
% =========================================================================
% Compute Q-Ball.
% 
% INPUT
% -------------------------------------------------------------------------
% fname_data				string
% fname_bvecs				string
% (opt)
%	fname_mask				string
%   fname_output			string. File output name. Default=name of the method
%	method					'qball'* | 'dti' | 'dsi'.  qball (Descoteaux), 'csd' (constrained spherical deconvolution, Tournier)
%	order_qball				2 | 4* | 6 | 8. order of SH decomposition. Default=4.
%	lambda					float. regularization parameter (default is 0.006)
%	sharpening				float. sharpening of the Q-Ball ODF . N.B. Only used for the Descoteaux method. Put 0 for no sharpening.
%	fname_log				string
% -------------------------------------------------------------------------
% 
% OUTPUT
% -------------------------------------------------------------------------
% -
% -------------------------------------------------------------------------
% 
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2010-02-23
% 2011-10-16: Modifs.
% 2011-11-06: Esthetic modifs.
% 
% =========================================================================



% INITIALIZATION
sh_order		= [0 6 0 15 0 28 0 45 0 0 0 91 0 0 0 153]; % nb of coefficients given the max order of SH decomposition
dbstop if error; % debug if error
if ~exist('opt'), opt = []; end
if ~isfield(opt,'fname_log'), opt.fname_log = 'log_j_dmri_compute_qball.txt'; end
if ~isfield(opt,'method'), opt.method = 'qball'; end
if ~isfield(opt,'fname_mask'), ismask=0, else ismask=1, end
if ~isfield(opt,'fname_output'), opt.fname_output = opt.method; end
if ~isfield(opt,'order_qball'), opt.order_qball = 4; end
if ~isfield(opt,'lambda'), opt.lambda = 0.006; end
if ~isfield(opt,'sharpening'), opt.sharpening = 0; end


% START FUNCTION
j_disp(opt.fname_log,['\n\n\n=========================================================================================================='])
j_disp(opt.fname_log,['   Running: j_dmri_compute_qball'])
j_disp(opt.fname_log,['=========================================================================================================='])
j_disp(opt.fname_log,['.. Started: ',datestr(now)])


% Check parameters
j_disp(opt.fname_log,['\nCheck parameters:'])
j_disp(opt.fname_log,['.. File data:           ',fname_data])
j_disp(opt.fname_log,['.. File bvecs:          ',fname_bvecs])
j_disp(opt.fname_log,['.. File mask:           ',opt.fname_mask])
j_disp(opt.fname_log,['.. File output:         ',opt.fname_output])
j_disp(opt.fname_log,['.. method:              ',opt.method])
j_disp(opt.fname_log,['.. order_qball:		',num2str(opt.order_qball)])
j_disp(opt.fname_log,['.. lambda:              ',num2str(opt.lambda)])
j_disp(opt.fname_log,['.. sharpening:		',num2str(opt.sharpening)])

% load data
j_disp(opt.fname_log,['\nLoad data...'])
j_disp(opt.fname_log,['.. File: ',fname_data])
data = read_avw(fname_data);

% load bvecs
j_disp(opt.fname_log,['\nLoad bvecs...'])
j_disp(opt.fname_log,['.. File: ',fname_bvecs])
bvecs = textread(fname_bvecs);

% get data dimensions
j_disp(opt.fname_log,['\nGet dimensions of the data...'])
nx = size(data,1);
ny = size(data,2);
nz = size(data,3);
nt = size(data,4);
j_disp(opt.fname_log,['.. Dimensions: ',num2str(nx),' x ',num2str(ny),' x ',num2str(nz),' x ',num2str(nt)])

% find where are the b=0 images
j_disp(opt.fname_log,['\nRemove b=0 images...'])
index_b0 = find(sum(bvecs,2)==0);
index_dwi = find(sum(bvecs,2)~=0);
j_disp(opt.fname_log,['.. Index of b=0 images: ',num2str(index_b0')])
nb_b0 = length(index_b0);
j_disp(opt.fname_log,['.. Number of b=0 images: ',num2str(nb_b0)])
nb_dwi = length(index_dwi);
j_disp(opt.fname_log,['.. Number of directions: ',num2str(nb_dwi)])

% load mask
if ismask
	j_disp(opt.fname_log,['\nLoad mask...'])
	j_disp(opt.fname_log,['.. File: ',opt.fname_mask])
	mask = read_avw(opt.fname_mask);
else
	j_disp(opt.fname_log,['\nNo masking. Use all voxels of the volume.'])
	mask = ones(nx,ny,nz);
end

% Reshape data
j_disp(opt.fname_log,['\nReshape data...'])
mask2d = logical(reshape(mask,1,nx*ny*nz));
data_2d = reshape(data,nx*ny*nz,nt);
clear data
data_dwi2d_masked = data_2d(mask2d,index_dwi);
% index_mask = find(mask2d);
% data_dwi2d = double(reshape(data_dwi,nx*ny*nz,nb_dwi));
% data_dwi2d_masked = data_dwi2d(mask2d,:);
nb_voxels = size(data_dwi2d_masked,1);
j_disp(opt.fname_log,['.. Number of voxels: ',num2str(nb_voxels)])

% % flip bvecs
% bvecs_temp = bvecs;
% for i=1:nb_dir
% 	switch (orientation)
% 		
% 	case 'coronal'
% 		bvecs_temp(i,1) = bvecs(i,1);
% 		bvecs_temp(i,2) = bvecs(i,3);
% 		bvecs_temp(i,3) = bvecs(i,2);
% 	end
% end
% bvecs = bvecs_temp;

% estimate ODF
switch(opt.method)

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

	% allocate memory
	j_disp(opt.fname_log,['\nAllocate memory...'])
	gfa2d = zeros(nb_voxels,1);
	qball2d = zeros(nb_voxels,sh_order(opt.order_qball));
	qball2d_tmp = zeros(nx*ny*nz,sh_order(opt.order_qball));
	gfa = zeros(nx,ny,nz);
	qball = zeros(nx,ny,nz,sh_order(opt.order_qball));

	% Define samplings for inputOrientationSet for order
	j_disp(opt.fname_log,['\nDefine sampling for SH order...'])
	scheme_dw = j_gen_scheme(bvecs, opt.order_qball);

	% Compute Q-Ball
	j_progress('Compute Q-Ball...')
	for i=1:nb_voxels
		S = data_dwi2d_masked(i,:);
		S_SH = j_amp2SH(S',scheme_dw);
		[qball_tmp gfa_tmp] = j_estimate_qball(S_SH,opt.order_qball,opt.lambda,opt.sharpening);
		gfa2d(i,1) = gfa_tmp;
		qball2d(i,:) = qball_tmp;
		j_progress(i/nb_voxels)
	end

	% reshape
	j_disp(opt.fname_log,['\nReshape...'])
	gfa(mask2d) = gfa2d;
	qball2d_tmp(mask2d,:) = qball2d;
	qball = reshape(qball2d_tmp,nx,ny,nz,sh_order(opt.order_qball));
	clear qball2d_tmp qball2d

	% Save files
	j_disp(opt.fname_log,['\nSave GFA...'])
	fname_gfa = [opt.fname_output,'_gfa'];
	save_avw(gfa,fname_gfa,'d',[nx ny nz]);
	j_disp(opt.fname_log,['Copy geometric information from ',fname_data,'...'])
	cmd = ['fslcpgeom ',fname_data,' ',fname_gfa,' -d'];
	j_disp(opt.fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	
	j_disp(opt.fname_log,['\nSave Q-Ball...'])
	fname_qball = [opt.fname_output];
	save_avw(qball,fname_qball,'d',[nx ny nz sh_order(opt.order_qball)]);

	
% 	gzip([fname_gfa,'.nii']);
% 	delete([fname_gfa,'.nii']);
% 	gzip([fname_qball,'.nii']);
% 	delete([fname_qball,'.nii']);
	
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

% END FUNCTION
j_disp(opt.fname_log,['\n.. Ended: ',datestr(now)])
j_disp(opt.fname_log,['==========================================================================================================\n'])


% % Define samplings for visu
% scheme_visu = gen_scheme(sampling_odf,order_visu);
% % display ODF
% if ~nx_odf, nx_odf=nx_mask; end
% if ~ny_odf, ny_odf=ny_mask; end
% 
% if strcmp(method,'dti')
% 	h_fig = j_display_odf(tensor,'coord','tensor','scheme',scheme_visu,'nx',nx_odf,'ny',ny_odf,'normalize',1,'overlay_gfa',1);
% else
% 	h_fig = j_display_odf(odf,'coord','sh','scheme',scheme_visu,'gfa',gfa,'nx',nx_odf,'ny',ny_odf);
% end
% 
% % save figure
% set(h_fig,'PaperPositionMode','auto')
% switch(method)
% 	case 'dti'
% 	print(h_fig,'-dpng','-r200',['fig_odf_',prefixe_mask,method,'_lODF',num2str(order_qball),'_sharp',num2str(sharpening),'_lambda',num2str(lambda),'_r300.png']);
% 
% 	case 'qball'
% 	print(h_fig,'-dpng','-r200',['fig_odf_',prefixe_mask,method,'_lODF',num2str(order_qball),'_sharp',num2str(sharpening),'_lambda',num2str(lambda),'_r300.png']);
% 
% 	case 'csd'
% 	print(h_fig,'-dpng','-r200',['fig_odf_',prefixe_mask,method,'_lDWI',num2str(order_dwi),'_lODF',num2str(order_csd),'_lVISU',num2str(order_visu),'.png']);
% end
