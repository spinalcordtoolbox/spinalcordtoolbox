function disco = j_distortion_correction_reversedGradient_v4(disco)
% =========================================================================
% 
% Distortion correction of EPI data using the reversed gradient method. 
% 
% Required data: two EPI with phase-encoding in the opposite directions (e.g., P-A and A-P). A deformation field will be computed
% form these two data, then applied to your fMRI or DTI data.
% 
% References:
%   Voss, Magnetic Resonance Imaging 24 (2006) 231?239
%   Cohen-Adad, NeuroImage 57 (2011) 1068?1076
%   Holland, Neuroimage 50 (2010) 175?183
% 
% INPUTS
% -----------------------------------
% disco                   structure
%   correct_intensity     = 1; % do intensity correction (r parameter)
%   thresh_quantile       = 50; % in percent
%   interpolate           = 4;
%   smoothing             = 3; % 3-D smoothing of the deformation field.
%   dilate_mask           = 5; % put size of dilation. Put 0 for no dilation.
%   interp_method         = 'linear';
%
% OUTPUT
% -----------------------------------
% disco                   structure
% 
% Author: Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-07-11
% 2012-01-08: (v3) Issue with permutation
% 2012-11-21: fixed semantic
% 2012-12-08: (v4) (i) Manage the case whre the user enters even number for gaussian smoothing. (ii) save deformation fields as nifti files.
% 
% =========================================================================


% globals
global interpolate
global interp_method

interpolate = disco.interpolate;
interp_method = disco.interp_method;

% Estimate transformation
fprintf('\nESTIMATE TRANSFORMATION\n');

% build file names
j_progress('Build file names ..............................');
if disco.switchPlusMinus
	fname_plus = disco.fname_minus;
	fname_minus = disco.fname_plus;
else
	fname_plus = disco.fname_plus;
	fname_minus = disco.fname_minus;
end
j_progress(1)

% check if files exist
if ~exist(fname_plus)==2 | ~exist(fname_minus)==2
	disp('ERROR: Check your input file names.');
	return
end

% load data
j_progress('Load data .....................................');
[data_plus,dims,scales,bpp,endian] = read_avw(fname_plus);
if ~sum(data_plus(:))
	disp('ERROR: check your file names')
	return;
end
data_minus = read_avw(fname_minus);
if ~sum(data_minus(:))
	disp('ERROR: check your file names')
	return;
end
[nx ny nz] = size(data_plus);
j_progress(1)

% use a mask?
if isempty(disco.fname_mask_plus)
	fname_mask_plus = '';
	fname_mask_minus = '';
else
	% get file name
	fname_mask_plus = disco.fname_mask_plus;
	fname_mask_minus = disco.fname_mask_minus;
	% check if files exist
	if ~exist(fname_mask_plus)==2 | ~exist(fname_mask_minus)==2
		disp('ERROR: Check your mask file names.');
		return
	end
	% load first mask
	if ~isempty(fname_mask_plus)
		[data_mask_plus hdr_mask_plus] = read_avw(fname_mask_plus);
		% load second mask
		if ~isempty(fname_mask_minus)
			[data_mask_minus] = read_avw(fname_mask_minus);
		else
			data_mask_minus = data_mask_plus;
		end
	else
		data_mask_plus = 0;
		data_mask_minus = 0;
	end
	% dilate masks
	if disco.dilate_mask
		j_progress('Dilate masks ..................................');
		data_mask_plus = j_morpho(data_mask_plus,'dilate',disco.dilate_mask,'disk');
		j_progress(0.5)
		data_mask_minus = j_morpho(data_mask_minus,'dilate',disco.dilate_mask,'disk');
		j_progress(1)
	end
	% mask data
	j_progress('Mask data .....................................');
	if sum(data_mask_plus(:))
		data_plus = data_plus.*data_mask_plus;
		data_minus = data_minus.*data_mask_minus;
	end
	j_progress(1)
end

% permute data (to get y dimension as phase encoding)
data_plus = permute(data_plus,disco.permute_data);
data_minus = permute(data_minus,disco.permute_data);
[nx ny nz] = size(data_plus);

% if user specified only one slice (for test purpose)
if disco.slice_numb
	nmin = disco.slice_numb;
	nmax = disco.slice_numb;
else
	nmin = 1;
	nmax = nz;
end

% estimate transformation
j_progress('Estimate transformation .......................');
yshift = zeros(nx,ny*interpolate,nz);
r = zeros(nx,ny*interpolate,nz);
% loop over slices
for iz=nmin:nmax

	% extract slice
	data_plus2d = data_plus(:,:,iz);
	data_minus2d = data_minus(:,:,iz);

	% remove noisy edges
	data_plus2dm = remove_background(data_plus2d,disco.thresh_quantile);
	data_minus2dm = remove_background(data_minus2d,disco.thresh_quantile);

	% estimate 2D-matrix for geometric distortion correction
	[yshift(:,:,iz) r(:,:,iz)] = estimate_transformation(data_plus2dm,data_minus2dm);

	% display progress
	j_progress((iz-nmin+1)/(nmax-nmin+1))
end


% Smooth deformation field (3D)
if disco.smoothing
	% check if DISCO.SMOOTHING is an odd number. 
	if mod(disco.smoothing+1,2) ~= 0
		% it is even
		disco.smoothing = round(disco.smoothing+1);
		disp(['Warning: DISCO.SMOOTHING was not an odd number --> new value: ',num2str(disco.smoothing)])
	end		
	j_progress('Smooth deformation field (3D) .................');
	yshift = smooth3(yshift,'gaussian',disco.smoothing);
	r = smooth3(r,'gaussian',disco.smoothing);
	j_progress(1)
end

% save deformation field
disco.yshift				= yshift;
disco.r						= r;

% save as deformation as nifti files
if disco.save_deformation
	
	% deformation field
	disp('Write deformation field ...');
	% subsample data
	j_progress('Subsample data ................................')
	yshift_sub = zeros(nx,ny,nz);
	r_sub = zeros(nx,ny,nz);
	for iz = 1:nz
		yshift2d = yshift(:,:,iz);
		r2d = r(:,:,iz);
		yshift2d_sub = zeros(nx,ny);
		r2d_sub = zeros(nx,ny);
		y = (1:1:ny);
		yi = (1/interpolate:1/interpolate:ny);
		yshift2d_sub = interp1(yi,yshift2d',y,interp_method,0)';
		r2d_sub = interp1(yi,r2d',y,interp_method,0)';
		yshift_sub(:,:,iz) = yshift2d_sub;
		r_sub(:,:,iz) = r2d_sub;
		j_progress(iz/nz)
	end	
	% permute data
	yshift_sub = permute(yshift_sub,[disco.permute_data 4]);
	r_sub = permute(r_sub,[disco.permute_data 4]);
	% write data
	save_avw_v2(yshift_sub,['deformation_geometric'],'d',scales);
	save_avw_v2(r_sub,['deformation_intensity'],'d',scales);
	% copy geometry information
	j_progress('Copy header from original data ................')
	cmd = ['fslcpgeom ',[disco.fname_plus],' deformation_geometric'];
	[status result] = unix(cmd); if status, error(result); end
	cmd = ['fslcpgeom ',[disco.fname_plus],' deformation_intensity'];
	[status result] = unix(cmd); if status, error(result); end	
	j_progress(1)
	disp('.. file created: deformation_geometric')
	disp('.. file created: deformation_intensity')

end



% =========================================================================
% apply transformation

fprintf('\nAPPLY TRANSFORMATION\n');
data_pluscorr = zeros(nx,ny,nz);
data_pluscorr_withoutr = zeros(nx,ny,nz);
j_progress('Apply transformation to data ..................');
for iz=nmin:nmax

	% extract slice
	data_plus2d = data_plus(:,:,iz);

	% apply geometric transformation
	[data_plus2dcorr data_plus2dcorr_withoutr] = apply_transformation(data_plus2d,yshift(:,:,iz),r(:,:,iz));	
	data_pluscorr_withoutr(:,:,iz) = data_plus2dcorr_withoutr;
	data_pluscorr(:,:,iz) = data_plus2dcorr;

	% display progress
	j_progress((iz-nmin+1)/(nmax-nmin+1))
end	

% display result
iz = disco.slice_numb;
if iz

	figure
	subplot(2,3,1)
	if isempty(disco.c_lim)
		c_lim = [min(min(data_plus(:,:,iz))) max(max(data_plus(:,:,iz)))];
	else
		c_lim = disco.c_lim;
	end
	imagesc(data_plus(:,:,iz),c_lim)
	if disco.flip_data, imagesc(flipud(data_plus(:,:,iz)),c_lim), end
	axis image, colorbar, grid
	title('T2+')

	subplot(2,3,4)
	if isempty(disco.c_lim)
		c_lim = [min(min(data_plus(:,:,iz))) max(max(data_plus(:,:,iz)))];
	else
		c_lim = disco.c_lim;
	end
	imagesc(data_minus(:,:,iz),c_lim)
	if disco.flip_data, imagesc(flipud(data_minus(:,:,iz)),c_lim), end
	axis image, colorbar, grid
	title('T2-')

	subplot(2,3,3)
	if isempty(disco.c_lim)
		c_lim = [min(min(data_plus(:,:,iz))) max(max(data_plus(:,:,iz)))];
	else
		c_lim = disco.c_lim;
	end
	imagesc(data_pluscorr_withoutr(:,:,iz),c_lim)
	if disco.flip_data, imagesc(flipud(data_pluscorr_withoutr(:,:,iz)),c_lim), end
	axis image, colorbar, grid
	title('T2+ corrected (without r)')

	subplot(2,3,6)
	if isempty(disco.c_lim)
		c_lim = [min(min(data_plus(:,:,iz))) max(max(data_plus(:,:,iz)))];
	else
		c_lim = disco.c_lim;
	end
	imagesc(data_pluscorr(:,:,iz),c_lim)
	if disco.flip_data, imagesc(flipud(data_pluscorr(:,:,iz)),c_lim), end
	axis image, colorbar, grid
	title('T2+ corrected')
% 	colormap cool

	subplot(2,3,2)
	imagesc(yshift(:,:,iz))
	if disco.flip_data, imagesc(flipud(yshift(:,:,iz))), end
	colorbar, grid
	title('yshift')

	subplot(2,3,5)
	imagesc(2./(1+r(:,:,iz)))
	if disco.flip_data, imagesc(flipud(2./(1+r(:,:,iz)))), end
	colorbar, grid
	title('r')
	
	disp_text = strcat('slice\_num = ',num2str(disco.slice_numb),', thresh\_quantile = ',num2str(disco.thresh_quantile),', smoothing\_window = ',num2str(disco.smoothing),', dilate\_mask = ',num2str(disco.dilate_mask));
	j_subplot_title(disp_text);
end

% correct or not for intensity distorsions
if ~disco.correct_intensity
	data_pluscorr = data_pluscorr_withoutr;
end

% write corrected data
disp('Write corrected data ...');
% if disco.permute_data
data_pluscorr = permute(data_pluscorr,[disco.permute_data 4]);
% end
save_avw_v2(data_pluscorr,[disco.fname_plus,disco.suffixe_output],'s',scales);
% copy geometry information
j_progress('Copy header from original data ................')
cmd = ['fslcpgeom ',fname_plus,' ',[disco.fname_plus,disco.suffixe_output]];
[status result] = unix(cmd); if status, error(result); end
j_progress(1)
disp(['.. file created: ',disco.fname_plus,disco.suffixe_output])




% =========================================================================
% Apply transformation on DW data
% =========================================================================
if disco.slice_numb
	fprintf('\n.. Only correct one slice (for debugging).\n\n');
	
elseif isempty(disco.fname_data)
	fprintf('\n.. No data provided (file name empty).\n\n');
	
else
	% build file name
	fname_data = disco.fname_data;
	
	% data get data dim
	j_progress('\nGet dimensions of the data ....................')
	cmd = ['fslsize ',fname_data];
	[status result] = unix(cmd);
	if status, error(result); end
	dims = j_mri_getDimensions(result);
	nx = dims(1);
	ny = dims(2);
	nz = dims(3);
	nt = dims(4);
	j_progress(1)
	disp(['-> ',num2str(nx),' x ',num2str(ny),' x ',num2str(nz),' x ',num2str(nt)])

	% load data
	data = zeros(nx,ny,nz,nt);
	[data,dims,scales,bpp,endian] = read_avw(fname_data);
	if ~sum(data(:)), disp('ERROR: Check diff data file name');	return; end
		
	% get number of volumes
	if disco.nbVolToDo
		nb_files = disco.nbVolToDo;
	else
		nb_files = nt;
	end
	disp(['-> Number of volumes to correct: ',num2str(nb_files)])
	
	% permute data (to get y dimension as phase encoding)
% 	if disco.permute_data
		data = permute(data,[disco.permute_data 4]);
% 	end
	[nx ny nz nt] = size(data);

	datacorr = zeros(nx,ny,nz,nt);
% 	datacorr_withoutr = zeros(nx,ny,nz,nt);

	% loop over slices
	for iT = 1:nb_files
		
		tmp = strcat(['Undistort data',' ',num2str(iT),'/',num2str(nb_files)]);
		disp_text = strcat(tmp,' ............................................');
		disp_text = disp_text(1:47);
		j_progress(disp_text);
		for iz=1:nz

			% extract slice
			data2d = data(:,:,iz,iT);

			% apply geometric transformation
			[data2dcorr] = apply_transformation(data2d,yshift(:,:,iz),r(:,:,iz));	
			datacorr(:,:,iz,iT) = data2dcorr;

			% display progress
			j_progress(iz/nz)
		end	
	end
	
	% write corrected data
	disp('Write distortion-corrected data ...')
% 	if disco.permute_data
	datacorr = permute(datacorr,[disco.permute_data 4]);
% 	end
	fname_data_disco = [fname_data,disco.suffixe_output];
	save_avw_v2(datacorr,fname_data_disco,'s',scales);
	
	% copy geometry information
	j_progress('Copy header from original data ................')
	cmd = ['fslcpgeom ',fname_data,' ',fname_data_disco];
	[status result] = unix(cmd); if status, error(result); end
	j_progress(1)
	disp(['.. file created: ',fname_data_disco])
	
end

disp('Distortion correction done.')











% =========================================================================
% function 'remove_background'
%
% This function mask noisy edges of images, based on the percent quantile
% of each linethe mean slice intensity.
% =========================================================================
function data2dm = remove_background(data2d,thresh_quantile);

% get dimensions
[nx ny] = size(data2d);

% retrieve nb of pixels
nb_pixels = nx*ny;

% convert data to 1D
data1d = reshape(data2d,1,nb_pixels);

% sort data regarding intensity
sorted_data = sort(data1d);

% find the intensity threshold knowing the quantile threshold
if thresh_quantile
	I_th = sorted_data(round(nb_pixels*thresh_quantile/100));
else
	I_th = 0;
end

% remove noisy voxels A EDGES ONLY (line independent)
% for iy=1:ny
% 	ix_min = min(find(data2d(:,iy)>I_th));
% 	ix_max = max(find(data2d(:,iy)>I_th));
% 	tmp = zeros(nx,1);
% 	tmp(ix_min:ix_max) = data2d(ix_min:ix_max,iy);
% 	data2dm(:,iy) = tmp;
% end

% remove all noisy voxels
filter_type     = 'gaussian'; % gaussian, average
hsize           = [5 5]; % for gaussian filter
sigma           = 0.5; % for gaussian filter
h = fspecial(filter_type,hsize,sigma);
data2df = imfilter(data2d,h,'replicate');
data2dm = data2df.*(data2d>I_th);
















% =========================================================================
% Geometric distorsion correction
% =========================================================================
function [yshift2d r2d] = estimate_transformation(data2d_plus,data2d_minus)

% globals
global interpolate
global interp_method
% global smoothing_window

% get dimensions
[nx ny] = size(data2d_plus);
yshift2d = zeros(nx,ny*interpolate);
r2d = zeros(nx,ny*interpolate);
y = (1:1:ny);
yi = (1/interpolate:1/interpolate:ny);

% loop over line
for ix=1:nx

	% initialization
	delta = ones(ny*interpolate,1);

	% retrieve phase encoding lines
	line_plus = data2d_plus(ix,:);
	line_minus = data2d_minus(ix,:);
% 	figure, plot(line_plus,'*'), hold on, plot(line_minus,'r*'), grid
	
	% compute cumulative intensity
	cumul_plus = cumsum(line_plus);
	cumul_minus = cumsum(line_minus);
% 	figure, plot(cumul_plus,'*'), hold on, plot(cumul_minus,'r*'), grid

	% check if cumulative function is null
	if ~sum(cumul_plus) | ~sum(cumul_minus)
		delta = zeros(ny*interpolate,1);
	else
		% normalization
		cumul_plus = cumul_plus/max(cumul_plus);
		cumul_minus = cumul_minus/max(cumul_minus);

		% interpolation
		cumul_plus = interp1(y,cumul_plus,yi,interp_method,0);
		cumul_minus = interp1(y,cumul_minus,yi,interp_method,0);

		% regularize interpolation result to get only positive value
		cumul_plus(find(cumul_plus<0)) = 0;
		cumul_minus(find(cumul_minus<0)) = 0;

		% transpose matrices
		cumul_plus_t = j_transpose(cumul_plus,1);
		cumul_minus_t = j_transpose(cumul_minus,1);
	end
		
	% compute the mean of both transposed matrices
	if delta
		delta = round(cumul_plus_t - (cumul_plus_t+cumul_minus_t)/2);
	end
	yshift2d(ix,:) = delta;

	% intensity correction
	line_plus = interp1(y,line_plus,yi,interp_method,0);
	line_minus = interp1(y,line_minus,yi,interp_method,0);
	for iy=1:ny*interpolate
		if iy+delta(iy)>0 & iy+delta(iy)<interpolate*ny+1
			i1 = line_plus(iy+delta(iy));
		else
			i1 = line_plus(iy);
		end
		if iy-delta(iy)>0 & iy-delta(iy)<interpolate*ny+1
			i2 = line_minus(iy-delta(iy));
		else
			i2 = line_minus(iy);
		end
		% check for null division
		if i1+i2
			i = 2.*i1*i2/(i1+i2);
		else
			i = 0;
		end
		if i2
			r2d(ix,iy) = i1/i2;
		else
			r2d(ix,iy) = 1;
		end
	end
end




% =========================================================================
% Apply transformation
% =========================================================================
function [data2d_corr data2d_corr_withoutr]= apply_transformation(data2d,yshift2d,r2d)

% globals
global interpolate
global interp_method

% get dimensions
[nx ny] = size(data2d);
data2d_corr = zeros(nx,ny);
data2d_corr_withoutr = zeros(nx,ny);
y = (1:1:ny);
yi = (1/interpolate:1/interpolate:ny);

% loop over line
for ix=1:nx

	% get line
	data1d = data2d(ix,:);
	
	% interpolate data
	data1di = interp1(y,data1d,yi,interp_method,0);
	
	% apply transformation
	for iy=1:ny*interpolate
		% geometric correction
		dy = round(yshift2d(ix,iy));
		if iy+dy>0 & iy+dy<interpolate*ny+1
			data1d_corr(iy) = data1di(iy+dy);
		else
			data1d_corr(iy) = data1di(iy);
		end
		% intensity correction
		data1d_corr_withoutr(iy) = data1d_corr(iy);
		data1d_corr(iy) = 2.*data1d_corr(iy)/(1+r2d(ix,iy));
	end
	
	% downsample data to original resolution
	data2d_corr_withoutr(ix,:) = downsample(data1d_corr_withoutr,interpolate);
	data2d_corr(ix,:) = downsample(data1d_corr,interpolate);
end



