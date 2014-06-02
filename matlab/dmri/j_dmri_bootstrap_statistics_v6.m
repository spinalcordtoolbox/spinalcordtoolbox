% =========================================================================
% FUNCTION
% dmri_bootstrap_statistics.m
%
% Process bootstrap data.
% N.B. Run dmri_bootstrap_generate() first!!!
% 
% COMMENTS
% Julien Cohen-Adad 2009-10-30
% 2012-03-15: use j_disp()
% 2012-03-26: fix bug: remove null ODF before computing JSD, etc.
% =========================================================================
function bootstrap = j_dmri_bootstrap_statistics_v6(bootstrap)



method_angularConfidence = 'minimizeAngleConstrained';
										  % 'minimizeAngle',
										  % 'maxOrdering': Simply order the maxima for both the ODF mean and the bootstrap ODF
										  % 'minimizeAngleConstrained': find the minimum cumulative angle for max1 and max2, without permitting both max from the mean ODF to get the same maximum from the bootstrap ODF


if isfield(bootstrap,'fname_log'), fname_log = bootstrap.fname_log; else fname_log = 'log_j_dmri_bootstrap_process_v6.txt'; end



% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_dmri_bootstrap_statistics_v6'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])



% get the angle of local maxima from the ODF
% j_cprintf('');
% j_cprintf('blue','\nCOMPUTE STATISTICS ON BOOTSTRAP DATA\n\n')
% max_cartesian = zeros(bootstrap.nb_bootstraps,3);
nx = bootstrap.nx;
ny = bootstrap.ny;
nq = bootstrap.sampling_visu;
bootstrap.nb_neighbours = round(nq/bootstrap.nb_neighb_ratio);
nb_bootstraps = bootstrap.nb_bootstraps;
% nq = dmri.dtk.sampling_odf;
% qball_odf = zeros(nq,nb_bootstraps);
% qball_mean = zeros(nx,ny,nz,nq);
% qball_std = zeros(nx,ny,nz,nq);
% qball_std_mean = zeros(nx,ny,nz);

% compile MEX files
if bootstrap.use_mex
	j_disp(fname_log,['\nGenerate MEX files...'])
	path_to_j_dmri_getMaxODF_mex = which('j_dmri_getMaxODF_mex2.c');
	mex(path_to_j_dmri_getMaxODF_mex);
	% path_to_j_dmri_getMaxODFconstrained_mex = which('j_dmri_getMaxODFconstrained_mex.c');
	% mex(path_to_j_dmri_getMaxODFconstrained_mex);
end

% generate scheme for visu and stat
bootstrap.scheme_visu = j_gen_scheme(nq,bootstrap.qball.order_odf);
scheme = bootstrap.scheme_visu;

% load colormap (for visu)
load('j_colormap_uncertainty','colormap_uncertainty')  

% Find each sample's neighbour on the sphere (used later to get the maxima of the ODF)
j_disp(fname_log,['\nFind each sample''s neighbour on the sphere...'])
vert = scheme.vert;
nb_samples=size(vert,1);
nearest_neighb = {};
distvect = [];
for isample=1:nb_samples
	% compute the euclidean distance between that guy and all other guys
	for i=1:nb_samples
		distvect = vert(isample,:)-vert(i,:);
		normdist(i) = norm(distvect);
	end
	% sort the distance in the increasing order
	[val nearest_neighb_tmp] = sort(normdist);
	% remove the first guy
	nearest_neighb_tmp = nearest_neighb_tmp(2:end);
	% save the first nb_neighbours
	nearest_neighb{isample} = nearest_neighb_tmp(1:bootstrap.nb_neighbours);
end
clear nearest_neighb_tmp
j_disp(fname_log,['.. Number of samples on the sphere: ',num2str(bootstrap.sampling_visu)])
j_disp(fname_log,['.. Number of neighbors: ',num2str(bootstrap.nb_neighbours)])

% load mask
mask = bootstrap.mask;

% Preallocation
% TODO: preallocate for jse and mse
% ip = 1;
odf_vox_mean = [];
odf_mean = zeros(nx,ny,nq);
odf_std = zeros(nx,ny,nq);
odf = zeros(nb_bootstraps,nq);
max_odf = {};
max_odf_mean = zeros(1,nq);
normdist = zeros(1,nq);
val = zeros(1,nq);
jsd_mean = zeros(nx,ny);
jsd_std = zeros(nx,ny);
blobiness = zeros(nx,ny);
rmse_mean = zeros(nx,ny);
rmse_std = zeros(nx,ny);
bootstrap.max_odf = [];
angular_error = zeros(1,nb_bootstraps);
angular_error2 = zeros(1,nb_bootstraps);
angular_err_mean = zeros(nx,ny);
angular_err_std = zeros(nx,ny);
angular_err_mean2 = zeros(nx,ny);
angular_err_std2 = zeros(nx,ny);

% loop over voxels
if ~bootstrap.display_stuff
%    j_progress('Compute statistics on the ODF .................')
end
for ix=1:nx
	for iy=1:ny		
		if bootstrap.display_stuff, fprintf('\nVOXEL %i/%i\n\n',ip,nx*ny); end

		if mask(ix,iy)

			for ib = 1:nb_bootstraps

				% Get ODF in SH coordinates
				odf_sh = bootstrap.qball.odf_sh(ib,ix,iy,:);
				% Convert it to amplitudes on the sphere (in cartesian)
				odf(ib,:) = j_SH2amp(squeeze(odf_sh),bootstrap.scheme_visu);
			end

			% Test for null ODF
%            j_disp(fname_log,['\nTest for null ODF'])
			null_odf = [];
            nonnull_odf = [];
			for ib=1:nb_bootstraps
				if ~sum(odf(ib,:))
					null_odf = cat(2,null_odf,ib);
                else
					nonnull_odf = cat(2,nonnull_odf,ib);                    
				end
            end
            if ~isempty(null_odf)
                j_disp(fname_log,['.. Voxel (',num2str(ix),',',num2str(iy),') --> Null ODFs: ',num2str(null_odf)])
            end
            
            % Remove null ODFs
            odf = odf(nonnull_odf,:);
            
 			% compute mean and STD of ODF
			if bootstrap.display_stuff
				j_progress('Compute mean and STD of ODF ...................')
			end
			odf_vox_mean = mean(odf,1);
			odf_mean(ix,iy,:) = odf_vox_mean;
			odf_std(ix,iy,:) = std(odf,1);
			if bootstrap.display_stuff, j_progress(1); end

       
%			if ~null_odf
				
				% Jensen-Shannon divergence between the mean ODF and each BS ODF
				if bootstrap.display_stuff
					j_progress('Compute Jensen-Shannon divergence .............')
				end
	% 			% normalize ODF
	% 			odf_vox_mean_jsd = (odf_vox_mean - min(odf_vox_mean))/max(odf_vox_mean - min(odf_vox_mean));
	%             odf_vox_mean_jsd = (odf_vox_mean_jsd) + 1000;
	% 			odf_jsd = odf;
	% 			for iBoot = 1:nb_bootstraps
	% 				odf_jsd(iBoot,:) = (odf(iBoot,:) - min(odf(iBoot,:)))/max(odf(iBoot,:) - min(odf(iBoot,:)));
	%                 odf_jsd(iBoot,:) = (odf_jsd(iBoot,:)) + 1000;
	% 			end
				% compute JSD
				jsd = j_stat_jensenShannonDivergence(odf,odf_vox_mean);
				jsd_mean(ix,iy) = mean(jsd);
				jsd_std(ix,iy) = std(jsd);
				if bootstrap.display_stuff, j_progress(1); end

				% compute blobiness
				blobiness(ix,iy) = max(odf_vox_mean)/min(odf_vox_mean);


	% 			% compute root mean sum of square between the mean and each BS matrix
	% 			if bootstrap.display_stuff
	% 				j_progress('Compute compute root mean sum of square .......')
	% 			end
	% 			for ib = 1:nb_bootstraps
	% 				distance_mat(ib) = (1/nq)*sqrt(sum(odf_vox_mean-odf(ib,:)).^2);
	% 			end
	% 			rmse_mean(ix,iy) = mean(distance_mat);
	% 			rmse_std(ix,iy) = std(distance_mat);
	% 			if bootstrap.display_stuff, j_progress(1); end


				% -------------------------------------------------------------
				%   MAXIMA
				% -------------------------------------------------------------

				% Get maxima of the mean ODF
	% 			use_mex = 1;
				if bootstrap.use_mex
					[max_odf_mean nb_maxima_mean_vox] = j_dmri_getMaxODF_mex2(squeeze(odf_mean(ix,iy,:))',scheme.vert,nearest_neighb,double(bootstrap.nb_neighbours),double(bootstrap.maxima_angular_threshold),double(bootstrap.maxima_amplitude_threshold));
				else
					[max_odf_mean nb_maxima_mean_vox] = j_dmri_getMaxODF(squeeze(odf_mean(ix,iy,:))',scheme,'nearest_neighb',nearest_neighb,'output','parametric');
				end
				% Get maxima of each bootstrap ODF
				if bootstrap.display_stuff
					j_progress('Find maxima of each bootstrap ODF .............')
                end
                nb_bootstraps_temp = size(odf,1);
				for ib = 1:nb_bootstraps_temp
					if bootstrap.use_mex
						[max_odf{ib} nb_maxima_vox(ib)] = j_dmri_getMaxODF_mex2(odf(ib,:),scheme.vert,nearest_neighb,double(bootstrap.nb_neighbours),double(bootstrap.maxima_angular_threshold),double(bootstrap.maxima_amplitude_threshold));
					else
						[max_odf{ib} nb_maxima_vox(ib)] = j_dmri_getMaxODF(odf(ib,:),scheme,'nearest_neighb',nearest_neighb,'output','parametric');
					end
					if bootstrap.display_stuff, j_progress(ib/nb_bootstraps); end

				end

				% Compute angular error on the first maxima of the ODF
				i_maxima = 1;
				if bootstrap.display_stuff
					j_progress('Compute angular error on the first maxima .....')
				end
	% 			figure
				ib2 = 1;
				angular_error = [];
				angular_error2 = [];

				% remove bootstrap sample with no maximum (it could actually happen)
				max_odf_new = {};
				iMax = 1;
				for ib = 1:nb_bootstraps_temp
					if nb_maxima_vox(nonnull_odf(ib))
						max_odf_new{iMax} = max_odf{ib};
						iMax = iMax+1;
					end
				end
				max_odf = max_odf_new;

				switch(method_angularConfidence)

					case('minimizeAngle')


					for ib = 1:length(max_odf_new)
		% 				angular_error_tmp2 = [];

						% loop across all maxima from the bootstrap ODF to select
						% the one with the lowest angle with the mean ODF first
						% maximum
						angular_error_tmp = [];
						nb_max_bootstrap = size(max_odf{ib},1);
						for imax = 1:nb_max_bootstrap

							% compute the dot product for each direction and take the inverse
							% cosine to get the angle IN DEGREE between the two vectors.
							% NB: the norm of each vector is assumed to be 1.
							angular_error_tmp(imax) = abs(acos(max_odf_mean(1,:)*max_odf{ib}(imax,:)'))*180/pi;
							% check if the selected maxima is inverted compared to the mean
							% maxima. If it is the case, then reverse it.
							if angular_error_tmp(imax) > bootstrap.angular_threshold
								angular_error_tmp(imax) = abs(acos(max_odf_mean(1,:)*(-max_odf{ib}(imax,:)')))*180/pi;
							end

						end
						% now take the min angle
						angular_error(ib) = min(angular_error_tmp);

						% Do the same stuff for the 2nd maxima of the ODF,
						% providing it exists.
						if size(max_odf_mean,1) >= 2
							% loop across all maxima from the bootstrap ODF to select
							% the one with the lowest angle with the mean ODF
							% second maximum
							angular_error_tmp = [];
							nb_max_bootstrap = size(max_odf{ib},1);
							for imax = 1:nb_max_bootstrap

								% compute the dot product for each direction and take the inverse
								% cosine to get the angle IN DEGREE between the two vectors.
								% NB: the norm of each vector is assumed to be 1.
								angular_error_tmp(imax) = abs(acos(max_odf_mean(2,:)*max_odf{ib}(imax,:)'))*180/pi;
								% check if the selected maxima is inverted compared to the mean
								% maxima. If it is the case, then reverse it.
								if angular_error_tmp(imax) > bootstrap.angular_threshold
									angular_error_tmp(imax) = abs(acos(max_odf_mean(2,:)*(-max_odf{ib}(imax,:)')))*180/pi;
								end

							end
							% now take the min angle
							angular_error2(ib) = min(angular_error_tmp);
						else
							angular_error2(ib) = 0;
						end

						% Now compute uncertainty for the 2nd direction, but only
						% if this one exists!
						if bootstrap.display_stuff, j_progress(ib/length(max_odf_new)); end
					end % ib


				case('maxOrdering')

					for ib = 1:length(max_odf_new)

						% compute the dot product for each direction and take the inverse
						% cosine to get the angle IN DEGREE between the two vectors.
						% NB: the norm of each vector is assumed to be 1.
						angular_error_tmp = abs(acos(max_odf_mean(1,:)*max_odf{ib}(1,:)'))*180/pi;
						% check if the selected maxima is inverted compared to the mean
						% maxima. If it is the case, then reverse it.
						if angular_error_tmp > bootstrap.angular_threshold
							angular_error_tmp = abs(acos(max_odf_mean(1,:)*(-max_odf{ib}(1,:)')))*180/pi;
						end
						angular_error(ib) = angular_error_tmp;

						% Do the same stuff for the 2nd maxima of the ODF,
						% providing it exists.
						if size(max_odf_mean,1) >= 2
							if size(max_odf{ib},1) >=2
								angular_error_tmp = abs(acos(max_odf_mean(2,:)*max_odf{ib}(2,:)'))*180/pi;
								% check if the selected maxima is inverted compared to the mean
								% maxima. If it is the case, then reverse it.
								if angular_error_tmp > bootstrap.angular_threshold
									angular_error_tmp = abs(acos(max_odf_mean(2,:)*(-max_odf{ib}(2,:)')))*180/pi;
								end
							else
								angular_error_tmp = 0;
							end
							angular_error2(ib) = angular_error_tmp;
						end

						if bootstrap.display_stuff, j_progress(ib/length(max_odf_new)); end
					end % ib				



				case('minimizeAngleConstrained')

					for ib = 1:length(max_odf_new)

						% compute the dot product for each direction and take the inverse
						% cosine to get the angle IN DEGREE between the two vectors.
						% NB: the norm of each vector is assumed to be 1.
						angular_error_tmp11 = abs(acos(max_odf_mean(1,:)*max_odf{ib}(1,:)'))*180/pi;
						% check if the selected maxima is inverted compared to the mean
						% maxima. If it is the case, then reverse it.
						if angular_error_tmp11 > bootstrap.angular_threshold
							angular_error_tmp11 = abs(acos(max_odf_mean(1,:)*(-max_odf{ib}(1,:)')))*180/pi;
						end

						% Do the same stuff for the 2nd maxima of the ODF,
						% providing it exists.
						if size(max_odf_mean,1) >= 2
							if size(max_odf{ib},1) >=2

								% Compute angular_error for the first maximum using the 2nd maximum of the bootstrap ODF	
								angular_error_tmp12 = abs(acos(max_odf_mean(1,:)*max_odf{ib}(2,:)'))*180/pi;
								if angular_error_tmp12 > bootstrap.angular_threshold
									angular_error_tmp12 = abs(acos(max_odf_mean(1,:)*(-max_odf{ib}(2,:)')))*180/pi;
								end
								% Compute angular_error for the 2nd maximum using the 1st maximum of the bootstrap ODF	
								angular_error_tmp21 = abs(acos(max_odf_mean(2,:)*max_odf{ib}(1,:)'))*180/pi;
								if angular_error_tmp21 > bootstrap.angular_threshold
									angular_error_tmp21 = abs(acos(max_odf_mean(2,:)*(-max_odf{ib}(1,:)')))*180/pi;
								end
								% Compute angular_error for the 2nd maximum using the 2nd maximum of the bootstrap ODF	
								angular_error_tmp22 = abs(acos(max_odf_mean(2,:)*max_odf{ib}(2,:)'))*180/pi;
								if angular_error_tmp22 > bootstrap.angular_threshold
									angular_error_tmp22 = abs(acos(max_odf_mean(2,:)*(-max_odf{ib}(2,:)')))*180/pi;
								end

								% minimize the cumulative angle
								sum1 = angular_error_tmp11+angular_error_tmp22;
								sum2 = angular_error_tmp12+angular_error_tmp21;
								if sum1<sum2
									angular_error(ib) = angular_error_tmp11;
									angular_error2(ib) = angular_error_tmp22;
								else
									angular_error(ib) = angular_error_tmp12;
									angular_error2(ib) = angular_error_tmp21;
								end
							else
								angular_error(ib) = angular_error_tmp11;						
								angular_error2(ib) = 0;
							end
						else
							angular_error(ib) = angular_error_tmp11;						
							angular_error2(ib) = 0;
						end

						if bootstrap.display_stuff, j_progress(ib/length(max_odf_new)); end
					end % ib
				end % switch


				% if there is more that one diff direction
	% 			if size(max_odf_mean,1) >= 2
	% 				% remove zeros from angular_error2
					angular_error2 = angular_error2(find(angular_error2));
					nb_bootstraps2 = length(angular_error2);
					if ~nb_bootstraps2, angular_error2=0; end
					% compute ang error mean and std
					angular_err_mean(ix,iy) = mean(angular_error);
					angular_err_std(ix,iy) = std(angular_error);
					nb_maxima(ix,iy) = nb_maxima_mean_vox;
					angular_err_mean2(ix,iy) = mean(angular_error2);
					angular_err_std2(ix,iy) = std(angular_error2);
					% compute confidence interval (P=0.05)
					if bootstrap.display_stuff
						j_progress('Compute confidence interval ...................')
					end
					% sort angular
					angular_err_sort = sort(angular_error);
					angular_err_sort2 = sort(angular_error2);
					% find the cut-off for 95% percentile
					angular_confidence(ix,iy) = angular_err_sort(round(length(max_odf_new)*(1-bootstrap.pvalue)));
					if nb_bootstraps2
						angular_confidence2(ix,iy) = angular_err_sort2(round(nb_bootstraps2*(1-bootstrap.pvalue)));
					else
						angular_confidence2(ix,iy) = 0;
					end
	% 			else
	% 				angular_err_mean2(ix,iy) = 0;
	% 				angular_err_std2(ix,iy) = 0;
	% 				% compute confidence interval (P=0.05)
	% 				if bootstrap.display_stuff
	% 					j_progress('Compute confidence interval ...................')
	% 				end
	% 				% sort angular
	% 				angular_err_sort = sort(angular_error);
	% 				% find the cut-off for 95% percentile
	% 				angular_confidence(ix,iy) = angular_err_sort(round(nb_bootstraps*(1-bootstrap.pvalue)));
	% 				angular_confidence2(ix,iy) = 0;
	% 			end
				if bootstrap.display_stuff, j_progress(1); end

				% display bootstrap ODF
	% 			odf_xy = reshape(odf(1:25,:),5,5,nq);
	% 			max_odf_xy = reshape(max_odf(1:25,:),5,5,nq);
	% 			j_display_odf(odf_xy,'max_odf',max_odf_xy);
	% 			j_display_odf(odf_vox_mean,'max_odf',max_odf_mean,'show_sampling',1);
	%  			j_display_odf(odf_mean,'max_odf',bootstrap.max_odf,'show_sampling',1);

				% save max odf structure
				bootstrap.max_odf{ix,iy} = max_odf_mean;
				bootstrap.max_odf_bootstrap{ix,iy,:} = max_odf;

% 			else
% 				fprintf('Null value.\n');
			end
			
% 		else
% 			if bootstrap.display_stuff
% 				fprintf('masked.\n');
% 			end
% 		end

		% update voxel number
% 		if ~bootstrap.display_stuff
% 			j_progress(ip/(nx*ny))
% 		end
% 		ip = ip+1;
	end
end


% pouf{1}=bootstrap.max_odf{14,8};
% j_display_odf(squeeze(odf_mean(14,8,:))','max_odf',pouf)
% 
% j_display_odf(odf_mean,'max_odf',bootstrap.max_odf,'normalize',0)

% save structure
j_progress('Save structure ................................')
gfa = squeeze(mean(bootstrap.qball.gfa,1));
bootstrap.gfa = gfa; % ADD MEAN!!!
bootstrap.odf_mean = odf_mean;
bootstrap.odf_std = odf_std;
% bootstrap.qball_std_mean = qball_std_mean;
bootstrap.jsd_mean = jsd_mean;
bootstrap.jsd_std = jsd_std;
bootstrap.blobiness = blobiness;
bootstrap.rmse_mean = rmse_mean;
bootstrap.rmse_std = rmse_std;
bootstrap.nb_maxima = nb_maxima;
bootstrap.angular_err_mean = angular_err_mean;
bootstrap.angular_err_std = angular_err_std;
bootstrap.angular_confidence = angular_confidence;
bootstrap.angular_err_mean2 = angular_err_mean2;
bootstrap.angular_err_std2 = angular_err_std2;
bootstrap.angular_confidence2 = angular_confidence2;
if ~exist(bootstrap.path), mkdir(bootstrap.path), end
save([bootstrap.path,bootstrap.file_struct,bootstrap.file_struct_suffixe],'bootstrap');
j_progress(1)



if bootstrap.display_figures
	
	j_progress('Display nice figures ..........................')

	% create image of GFA
	img = bootstrap.gfa;
	h_fig = figure; imagesc(img,bootstrap.fig.gfa_scaling); axis image, axis off; colormap gray;
	if bootstrap.fig.colorbar,
		colorbar;
		title('GFA');
	end
	print(h_fig,'-dpng','-r100',[bootstrap.path,'_fig_gfa',bootstrap.fig_suffixe,'.png']);

	j_progress(0.2)

	% create image of JSD
	img = abs(bootstrap.jsd_mean);
	h_fig = figure; imagesc(img,bootstrap.fig.jsd_scaling); axis image, axis off; colormap jet;
	if bootstrap.fig.colorbar,
		colorbar;
		title('Jensen-Shannon divergence');
	end
	print(h_fig,'-dpng','-r100',[bootstrap.path,'_fig_jsd',bootstrap.fig_suffixe,'.png']);
	j_progress(0.4)

	% create image of blobiness
	img = abs(bootstrap.blobiness);
	h_fig = figure; imagesc(img); axis image, axis off; colormap jet;
	if bootstrap.fig.colorbar,
		colorbar;
		title('Blobiness');
	end
	print(h_fig,'-dpng','-r100',[bootstrap.path,'_fig_blobiness',bootstrap.fig_suffixe,'.png']);
	j_progress(0.5)

	% create image of nb of maxima (from the mean data)
	img = bootstrap.nb_maxima;
	h_fig = figure; imagesc(img,[1 3]); axis image, axis off; colormap Autumn;
	if bootstrap.fig.colorbar,
		colorbar;
		title('Nb Maxima');
	end
	print(h_fig,'-dpng','-r100',[bootstrap.path,'_fig_nb_maxima',bootstrap.fig_suffixe,'.png']);
	j_progress(0.6)

	% create image of confidence interval for 1st ODF maxima
	img = squeeze(bootstrap.angular_confidence);
	h_fig = figure; imagesc(abs(img),[0 90]); axis image, axis off;
	load('j_colormap_uncertainty','colormap_uncertainty')
	set(h_fig,'Colormap',colormap_uncertainty);
	if bootstrap.fig.colorbar,
		colorbar;
		title('Confidence interval of the 1st maxima (95%)');
	end
	print(h_fig,'-dpng','-r100',[bootstrap.path,'_fig_angular_confidence',bootstrap.fig_suffixe,'.png']);
	j_progress(0.8)

	% create image of confidence interval for 2nd ODF maxima
	img = squeeze(bootstrap.angular_confidence2);
	h_fig = figure; imagesc(abs(img),[30 90]); axis image, axis off;
	set(h_fig,'Colormap',colormap_uncertainty);
	if bootstrap.fig.colorbar,
		colorbar;
		title('Confidence interval of the 2nd maxima (95%)');
	end
	print(h_fig,'-dpng','-r100',[bootstrap.path,'_fig_angular_confidence2',bootstrap.fig_suffixe,'.png']);
	j_progress(1)

	% create image of STD 2nd ODF maxima
	% img = squeeze(bootstrap.angular_confidence2);
	% h_fig = figure; imagesc(abs(img),[0 90]); axis image, axis off;
	% colorbar;

	if bootstrap.display_odf
		% create image of ODF maps
		h_fig = j_display_odf(bootstrap.odf_mean,'coord','amp','overlay',bootstrap.gfa);
		set(h_fig,'PaperPositionMode','auto');
		print(h_fig,'-dpng','-r300',[bootstrap.path,'_fig_odf',bootstrap.fig_suffixe,'.png']);

		% create image of maxima
		h_fig = j_display_odf(bootstrap.odf_mean,'coord','amp','overlay',bootstrap.gfa,'max_odf',bootstrap.max_odf,'max_color','fixed','linewidth',3,'show_odf',0);
		set(h_fig,'PaperPositionMode','auto');
		print(h_fig,'-dpng','-r300',[bootstrap.path,'_fig_maxima',bootstrap.fig_suffixe,'.png']);

		% create image of bootstrap maxima
		if bootstrap.nb_bootstraps<10
			h_fig = j_display_odf(bootstrap.odf_mean,'coord','amp','overlay',bootstrap.gfa,'max_odf',bootstrap.max_odf_bootstrap,'max_color','fixed','nb_bootstraps',bootstrap.nb_bootstraps,'linewidth',0.5,'show_odf',0);
		else
			h_fig = j_display_odf(bootstrap.odf_mean,'coord','amp','overlay',bootstrap.gfa,'max_odf',bootstrap.max_odf_bootstrap,'max_color','fixed','nb_bootstraps',10,'linewidth',0.5,'show_odf',0);
		end
		set(h_fig,'PaperPositionMode','auto');
		print(h_fig,'-dpng','-r300',[bootstrap.path,'_fig_maxima_bootstrap',bootstrap.fig_suffixe,'.png']);
	end


	if bootstrap.close_figures, close all; end
end


% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])








% =========================================================================
% FUNCTION
% reorient
% =========================================================================
function img_new = reorient(img,orientation)

switch(orientation)
	
	case 'axial'
		img_new = flipud(fliplr(img'));

end

