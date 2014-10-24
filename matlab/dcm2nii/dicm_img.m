function img = dicm_img(s)
% img = dicm_img(metaStructOrFilename);
% 
% DICM_IMG reads image from a dicom file.
% 
% The mandatory input is the dicom file name, or the struct returned by
% dicm_hdr. The output keeps the data type in dicom file.
% 
% DICM_IMG is like dicomread from Matlab, but is independent of Image Processing
% Toolbox. A major difference from dicomread is that DICM_IMG does not transpose
% the result image. This avoids transposing it again during nifti conversion,
% but the Columns and Rows parameters become counter-intuitive.
% 
% Limitation: DICM_IMG reads only grayscale image with little endian format,
% which seems to be the case for dicom from major vendors nowadays. It can deal
% with only JPEG compression.
%
% See also DICM_HDR DICM_DICT DICM2NII

% History (yyyy/mm/dd):
% 20130823 Write it for dicm2nii.m (xiangrui.li@gmail.com)
% 20130914 Use PixelData.Bytes rather than nPixels;
%          Use PixelRepresentation to determine signed data.
% 20130923 Use BitsAllocated for bits. Make it work for multiframe.
% 20131018 Add Jpeg de-compression part.

persistent flds dict;
if isempty(flds)
    flds = {'Columns' 'Rows' 'BitsAllocated'};
    dict = dicm_dict('', [flds {'SamplesPerPixel' 'PixelRepresentation'}]);
end
if isstruct(s) && ~all(isfield(s, flds)), s = s.Filename; end
if ~isstruct(s), s = dicm_hdr(s, dict); end % input is file name
if isempty(s), error('File not exist or not a dicom file.'); end
if isfield(s, 'SamplesPerPixel') && s.SamplesPerPixel>1
    error('SamplesPerPixel of greater than 1 not supported.'); 
end

fid = fopen(s.Filename);
if fid<0
    if exist([s.Filename '.gz'], 'file')
        gunzip([s.Filename '.gz']);
        fid = fopen(s.Filename);
    end
    if fid<0, error(['File not exists: ' s.Filename]); end
end
closeFile = onCleanup(@() fclose(fid));
fseek(fid, s.PixelData.Start, -1);
if ~isfield(s.PixelData, 'Format')
    fmt = sprintf('*uint%g', s.BitsAllocated);
else
    fmt =  s.PixelData.Format;
end

if ~isfield(s, 'TransferSyntaxUID') || ... % maybe PAR or AFNI file
        strcmp(s.TransferSyntaxUID, '1.2.840.10008.1.2.1') || ...
        strcmp(s.TransferSyntaxUID, '1.2.840.10008.1.2')
    n = s.PixelData.Bytes / double(s.BitsAllocated) * 8;
    img = fread(fid, n, fmt);
    dim = double([s.Columns s.Rows]);
    img = reshape(img, [dim n/dim(1)/dim(2)]); % 3rd dimension is frame
else % may not be always jpeg compression, just let imread err out
    b = fread(fid, inf, '*uint8'); % read all as bytes
    nEnd = numel(b) - 8; % terminator 0xFFFE E0DD and its zero length
    n = typecast(b(5:8), 'uint32'); i = 8+n; % length of offset table
    if n>0
        nFrame = n/4; % # of elements in offset table 
    else % empty offset table
        ind = strfind(b', uint8([254 255 0 224])); % 0xFFFE E000
        nFrame = numel(ind) - 1; % one more for offset table, even if empty
    end
    img = zeros(s.Rows, s.Columns, nFrame, fmt(2:end)); % pre-allocate
    fname = [tempname '.jpeg'];
    deleteTemp = onCleanup(@() delete(fname));
    for j = 1:nFrame
        i = i+4; % delimiter: FFFE E000
        n = typecast(b(i+uint32(1:4)), 'uint32'); i = i+4;
        fid = fopen(fname, 'w+');
        fwrite(fid, b(i+(1:n)), 'uint8'); i = i+n; % jpeg data
        fclose(fid); 
        img(:,:,j) = imread(fname); % take care of various jpeg compression
        if i>=nEnd % in case false delimiter in data was counted
            img(:,:,j+1:end) = [];
            break;
        end
    end
    img = permute(img, [2 1 3]); % result in dicom convention
end

if isfield(s, 'PixelRepresentation') && s.PixelRepresentation>0
    img = reshape(typecast(img(:), fmt(3:end)), size(img));  % signed
end
