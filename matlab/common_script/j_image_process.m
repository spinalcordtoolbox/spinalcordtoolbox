% =========================================================================
% FUNCTION
% j_image_process
%
% Process images.  To identify coordinates, open an image first using:
%   data = imread(fname_image); figure, imagesc(data); axis image
%
% INPUT
% (-)
%
% OUTPUT
% (-)
%
% COMMENTS
% Julien Cohen-Adad  <jcohen@nmr.mgh.harvard.edu>
% 2007-07-27: created
% 2012-01-11: modifs
% =========================================================================
function j_image_process(opt)


% Parameters
%===========================
% Todo
image_process   = [0 0 0 0 1 0 0]; % histogram normalization, 
								 % contrast adjustment
								 % resize
								 % filter
								 % crop
								 % rotate
								 % add black edges

% Output file
output_format   = 'png'; % 'jpg', 'png'
prefixe_write   = 'c_';
quality_index   = 90; % 0-100, 100 is the best quality (for jpg only)

% Resize
resize_scale    = 0.5;
resize_method   = 'bilinear'; % bilinear, bicubic

% Filter
filter_type     = 'gaussian'; % gaussian, average
hsize           = [4 4]; % for gaussian filter
sigma           = 0.5; % for gaussian filter

% Crop
x=1015;
y=456;
boxx=1378-x;
boxy=883-y;
crop_mask		= [x,x+boxx-1;y,y+boxy-1]; % xmin,xmax;ymin,ymax
flip_mask       = 1; 

% Rotate
rotate_angle    = 180; % in degree, should be multiple of 90

% add black edges
x=16;


% default initialization
if (nargin<0), help j_image_process; return; end
disp_text       = 1;
if ~exist('opt'), opt = []; end
if isfield(opt,'disp_text'), disp_text = opt.disp_text; end
if flip_mask, crop_mask = flipud(crop_mask); end

% get images
fprintf('\nLoad images...\n')
opt_getfiles.ext_filter = '*';
opt_getfiles.windows_title = 'Select JPG images';
opt_getfiles.file_selection = 'matlab';
opt_getfiles.output = 'cell';
fname_data = j_getfiles(opt_getfiles);
if ~iscell(fname_data), return, end
nt = size(fname_data,2);

% loop on each file
% if (disp_text), j_progress('Process images...'); end
fprintf('Process images...\n')
for i=1:nt
    % load data
    [data] = imread(fname_data{i});
    [nx ny nz] = size(data);
    
    % convert to double and normalize to range [0 1]
    data = im2double(data);
    
    % rescale image
    if image_process(3)
        data=imresize(data,resize_scale,resize_method);
        crop_mask_new = round(resize_scale.*crop_mask);
    else
        crop_mask_new = crop_mask;
    end
    
    % normalize histogram
    if image_process(1)
        for i_rgb=1:3
            data(:,:,i_rgb)=histeq(data(:,:,i_rgb));
        end
    end
    
    % adjust contrast
    if image_process(2)
        for i_rgb=1:3
            data(:,:,i_rgb)=imadjust(data(:,:,i_rgb));
        end
    end
    
    % smooth image
    if image_process(4)
        if strcmp(filter_type,'average')
            h = fspecial(filter_type,hsize);
        elseif strcmp(filter_type,'gaussian')
            h = fspecial(filter_type,hsize,sigma);
        end
        for i_rgb=1:3
            data_tmp(:,:,i_rgb) = imfilter(data(:,:,i_rgb),h,'replicate');
        end
        clear data
        data = data_tmp;
        clear data_tmp
    end
    
    % crop image
    if image_process(5)
        for i_rgb=1:3
            data_tmp(:,:,i_rgb) = data(crop_mask_new(1,1):crop_mask_new(1,2),crop_mask_new(2,1):crop_mask_new(2,2),i_rgb);
        end
        clear data
        data = data_tmp;
        clear data_tmp
    end
    
    % rotate image
    if image_process(6)
        for i_rgb=1:3
            data_tmp{i_rgb} = data(:,:,i_rgb);
            nb_rot = rotate_angle/90;
            for i_rot=1:nb_rot
                data_tmp{i_rgb} = rot90(data_tmp{i_rgb});
            end
        end
        clear data
        for i_rgb=1:3
            data(:,:,i_rgb) = data_tmp{i_rgb};
        end
        clear data_tmp
	end
	
	% add black edges
    if image_process(7)
        for i_rgb=1:3
            data_tmp(:,:,i_rgb) = cat(2,zeros(nx,x),data(:,:,i_rgb),zeros(nx,x));
        end
        clear data
        data = data_tmp;
        clear data_tmp
    end
	
 
    % write new image
    [path_read file_read ext_read] = fileparts(fname_data{i});
    ext_write = strcat('.',output_format);
    fname_write = strcat(path_read,filesep,prefixe_write,file_read,ext_write);
    warning off
    switch output_format
        case 'jpg'
        imwrite(data,fname_write,'Bitdepth',8,'Mode','lossy','Quality',quality_index);
        
        case 'png'
        imwrite(data,fname_write,'Bitdepth',8);
    end
    warning on
    
    % display progress  
    fprintf('-> generated file: %s\n',strcat(prefixe_write,file_read,ext_write));
%     if (disp_text), j_progress(i/nt); end
end
fprintf('\n')

% display output
% if (disp_text)
%     for i=1:nt
%         fprintf('-> generated file: %s\n',strcat(file_write(i,:)));
%     end
% end
