% =========================================================================
% FUNCTION
% j_mri_write
%
% Write MRI volumes.
% 
% Accept Analyze 7.5 and Nifti formats.
%
% INPUTS
% data              4D or 3D array. data(:,:,:,i) is the data of the ith file.
% header            structure. header(i) is a description of the .hdr and .mat
% file_mask         prefixe of the images to write (e.g. 'FILT')
% (argin)           structure with different optional fields:
%   disp_text       binary. By default, time is not displayed
%   text_to_display string.
%   wait_bar        binary. By default, there is no waitbar
%   path_save       new save path
%   origin          [x y z]. By default, origin is the volume center
%   rotate_volume   0,90,180,270 in trigo sense (default=0)
%   slices_order    'normal', 'inversed' (default='normal')
%   format_spm      binary. Default=1
%   norm_scale      binary. Default=0
%   output			'nii','nii_gz'
%
% OUTPUTS
% (-)
%
% COMMENTS
% Julien Cohen-Adad 2009-04-04
% =========================================================================
function j_mri_write(data,header,file_mask,varargin)


% initializations
if (nargin<2), help j_mri_write; return; end
if (nargin<3), file_mask = ''; end
if (nargin<4), argin=[]; else, argin=varargin{1}; end
% if isfield(argin,'file_mask'), file_mask=argin.file_mask; else file_mask=''; end
if isfield(argin,'disp_text'), disp_text=argin.disp_text; else disp_text=0; end
if isfield(argin,'wait_bar'), wait_bar=argin.wait_bar; else wait_bar=0; end
if isfield(argin,'path_save'), path_save=argin.path_save; else path_save=''; end
if isfield(argin,'rotate_volume'), rotate_volume=argin.rotate_volume; else rotate_volume=0; end
if isfield(argin,'slice_order'), slice_order=argin.slice_order; else slice_order='normal'; end
if isfield(argin,'origin'), origin=argin.origin; else origin=''; end
if isfield(argin,'text_to_display'), text_to_display=argin.text_to_display; else text_to_display='Writing data. Please wait...'; end
% if isfield(argin,'create_mat'), create_mat=argin.create_mat; else create_mat=0; end
if isfield(argin,'format_spm'), format_spm=argin.format_spm; else format_spm=1; end
if isfield(argin,'norm_scale'), norm_scale=argin.norm_scale; else norm_scale=0; end
if isfield(argin,'output'), output=argin.output; else output='nii'; end

% warning_struct = warning;
% warning_state = warning_struct(1).state;
% warning off

% get local parameters
j_parameters

nt = size(data,4);
spm_defaults;

if wait_bar, h = waitbar(0,'Writing data. Please wait...'); end
if disp_text, tic; fprintf(text_to_display); end

% flip volume if asked to do so
switch rotate_volume
case(0)
case(90)
    for iSlice=1:size(data,3)
        data(:,:,iSlice,:)=rot90(data(:,:,iSlice,:));
    end
case(180)
    for iSlice=1:size(data,3)
        data(:,:,iSlice,:)=rot90(data(:,:,iSlice,:),2);
    end
case(270)
    for iSlice=1:size(data,3)
        data(:,:,iSlice,:)=rot90(data(:,:,iSlice,:),3);
    end
end

% change slice order
if strcmp(slice_order,'inversed')
    for iSlice=1:size(data,3)
        data3d_tmp(:,:,iSlice)=(data(:,:,end-iSlice+1,:));
    end
    data = data3d_tmp;
    clear data3d_tmp;
end

% write header
for i = 1:nt
	
	switch param.spm.version
		
		case 'spm2'
		% change origin
		if exist('origin')
			header(i).private.hdr.hist.origin = origin;
		end

		if (wait_bar) waitbar(i/nt,h); end
		[path_read file_read ext_read] = fileparts(header(i).fname);
		if ~isempty(path_save)
			path_read = path_save;
		else
			path_read = strcat(path_read,filesep);
		end
		file_read_new = strcat(file_mask,file_read);
		fname_new = strcat(path_read,file_read_new,ext_read);
		header(i).fname = fname_new;

		% create folder if doesn't exist
		if ~isdir(path_read) mkdir(path_read); end

		% write volume
		if norm_scale
			j_spm_write_vol_normScale(header(i),data(:,:,:,i));
			delete(strcat(path_read,file_read_new,'.mat'));
		elseif ~format_spm
			j_spm_write_vol(header(i),data(:,:,:,i),origin);
			delete(strcat(path_read,file_read_new,'.mat'));
		else
			spm_write_vol(header(i),data(:,:,:,i));
		end
		
		
		
		case 'spm5'
% 		% change origin
% 		if exist('origin')
% 			header(i).private.hdr.hist.origin = origin;
% 		end

		if (wait_bar) waitbar(i/nt,h); end
		[path_read file_read ext_read] = fileparts(header(i).fname);
		if ~isempty(path_save)
			path_read = path_save;
		elseif isempty(path_read)
			path_read = '';
		else
			path_read = strcat(path_read,filesep);
		end
		file_read_new = strcat(file_mask,file_read);
		fname_new = strcat(path_read,file_read_new,ext_read);
		header(i).fname = fname_new;

		% create folder if doesn't exist
		if ~isempty(path_read)
			if ~isdir(path_read)
				mkdir(path_read);
			end
		end

		% write volume
		warning off % prevent warning related to FINITE command
		spm_write_vol(header(i),data(:,:,:,i));
		warning on
		
		% output format
		if strcmp(output,'nii_gz')
			gzip(fname_new);
			delete(fname_new);
		end
		
	end % switch
	clear fname_new;

end % i


if (disp_text) fprintf(' OK (elapsed time %s seconds) \n',num2str(toc)); end
if (wait_bar) close(h); end

% [path_write file_write ext_write] = fileparts(header.fname);
% file_write = strcat(file_write,ext_write);

% if strcmp(warning_state,'off')
%     warning off
% else
%     warning on
% end

