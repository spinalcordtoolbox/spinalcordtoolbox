function [s, info] = dicm_hdr(fname, dict, iFrames)
% [s, err] = dicm_hdr(dicomFileName, dict, iFrames);
% 
% DICM_HDR returns header of a dicom file in struct s.
% 
% The mandatory 1st input is the dicom file name. The optional 2nd input can be
% a dicom dict, which may have only part of the full dict. The partial dict can
% be returned by dict = dicm_dict(vendor, fieldNames). The use of partial dict
% may speed up header read considerably. See rename_dicm for example.
% 
% The optional 3rd intput is only needed for multi-frame dicom files. When there
% are many frames, it may be very slow to read all items in
% PerFrameFunctionalGroupsSequence for all frames. The 3rd input can be used to
% specify the frames to read. By default, items for only 1st, 2nd and last
% frames are read.
% 
% The optional 2nd output contains information in case of error, and will be
% empty if there is no error.
% 
% DICM_HDR is like dicominfo from Matlab, but is independent of Image Processing
% Toolbox. The limitation is it can deal with only little endian data for
% popular vendors. The advantage is that it decodes most private and shadow tags
% for Siemens, GE and Philips dicom, and runs faster, especially for partial
% header and multi-frame dicom.
% 
% This can also read Philips PAR file and AFNI HEAD file, and return needed
% fields for dicm2nii to convert into nifti.
% 
% See also DICM_DICT DICM2NII DICM_IMG RENAME_DICM

% The method used here:
% Check 4 bytes at 128 to make sure it is 'DICM';
% Find PixelData; Get its location;
% Loop through each item:
%      Read tag: group and element, each 1 uint16;
%        Find name in dictionary by the tag; if not exists,
%        assign it as Private_xxxx_xxxx;
%      Get VR:
%        Read VR (2 char) if explicit VR; 
%        Get VR from dict if implicit;
%      Decode item length type:
%        implicit VR, always uint32;
%        explicit VR: uint16/uint32(skip 2 bytes) based on VR; 
%      Decode data type by VR;
%      if VR == 'SQ', deal in special way;
%      Read the item according to the length and data type;
%        Process the item if needed;
%      Assign to field.

% History (yyyy/mm/dd):
% 20130823 Write it for dicm2nii.m (xiangrui.li@gmail.com).
% 20130912 Extend private tags, automatically detect vendor.
% 20130923 Call philips_par, so make dicm2nii easier. 
% 20131001 Decode SQ, useful for multiframe dicom and Philips Stack. 
% 20131008 Load then typecast. Faster than multiple fread.
% 20131009 Make it work for implicit VR.
% 20131010 Decode Siemens CSA header, so it is human readable.
% 20131019 PAR file: read image col labels, and use it for indexing.
% 20131023 Implement afni_hdr.
% 20131102 Use last tag for partial hdr, so return if it is non-exist fld.
% 20131107 Search tags if only a few fields: faster than regular way.
% 20131114 Add 3rd input: only 1,2,last frames hdr read. 0.4 vs 38 seconds!
%          Store needed fields in LastFile for PAR MIXED image type.
% 20140123 Support dicom without meta info (thanks Paul).
% 20140213 afni_head: IJK_TO_DICOM_REAL replaces IJK_TO_DICOM.
% 20140502 philips_par: don't use FOV for PixelSpacing and SpacingBetweenSlices.
% 20140506 philips_par: use PAR file name as SeriesDescription for nii name.
% 20140512 decode GE ProtocolDataBlock (gz compressed).
% 20140611 No re-do if there are <16 extra bytes after image data.
% 20140724 Ignore PAR/HEAD ext case; fix philips_par: Patient Position.
% 20140924 Use dict VR if VR==OB/UN (thx Macro R). Could be bad.

persistent dict_full;
s = []; info = '';
fullHdr = false;
if nargin<2 || isempty(dict)
    if isempty(dict_full), dict_full = dicm_dict; end
    fullHdr = true;
    dict = dict_full; 
end
if nargin<3, iFrames = []; end

fid = fopen(fname);
if fid<0, info = ['File not exists: ' fname]; return; end
cln = onCleanup(@() fclose(fid));
fseek(fid, 128, -1);
sig = fread(fid, 4, '*char')';
isDicm = strcmp(sig, 'DICM');
isTruncated = false;
if ~isDicm
    fseek(fid, 0, -1);
    foo = fread(fid, 1, 'uint16');
    if foo==2 || foo==8 % truncated dicom? not safe, but no better way
        fseek(fid, 0, -1);
        isTruncated = true;
    end
end
if ~isDicm && ~isTruncated % may be PAR or HEAD file
    [~, ~, ext] = fileparts(fname);
    if strcmpi(ext, '.PAR') || strcmpi(ext, '.REC')
        func = @philips_par;
    elseif strcmpi(ext, '.HEAD') || strcmpi(ext, '.BRIK')
        func = @afni_head;
    else
        info = ['Unknown file type: ' fname];
        return;
    end
    try [s, info] = feval(func, fname);
    catch me
        info = me.message;
        return;
    end
    if ~isempty(s), return; end
    info = ['Not dicom file: ' fname]; 
    return; 
end

% This is the trick to make partial hdr faster.
b = [];
PixelData = uint8([224 127 16 0]); % 0x 0010 7fe0, can't add VR
for nb = [120000 2e6 20e6 Inf] % if not enough, read more
    b = [b fread(fid, nb, '*uint8')']; %#ok
    allRead = feof(fid);
    i = strfind(b, PixelData);
    if ~isempty(i)
        break;
    elseif allRead
        info = ['No PixelData in ' fname]; 
        return; 
    end
end
s.Filename = fopen(fid);

% iPixelData could be in header or data. Using full hdr can correct this
iPixelData = i(end); % start of PixelData tag with 132 offset
b = typecast(b, 'uint16'); % hope this makes the code faster

i = 1; len = numel(b)-6; % 6 less avoid missing next tag
expl = false; explVR = false; % default for truncated dicom
toSearch = numel(dict.tag)<10;

if toSearch % search each tag if only a few fields
    b8 = typecast(b, 'uint8');
    tg = uint8([2 0 16 0 'UI']); % TransferSyntaxUID
    i = (strfind(b8, tg)+1) / 2; % i for uint16
    if ~isempty(i) % empty for truncated 
        [dat, name] = read_item(i(1));
        s.(name) = dat;
        expl = ~strcmp(dat, '1.2.840.10008.1.2');
    end
    for k = 1:numel(dict.tag)
        tg = typecast(dict.tag(k), 'uint8');
        tg = tg([3 4 1 2]);
        if expl, tg = [tg uint8(dict.vr{k})]; end %#ok safer with vr
        i = (strfind(b8, tg)+1) / 2;
        if isempty(i), continue;
        elseif numel(i)>1 % +1 tags found. use non-search method
            i = 1;
            toSearch = false;
            break; % re-do in regular way
        end
        [dat, name, info] = read_item(i);
        s.(name) = dat;
    end
end

while ~toSearch
    if i>=len
        if strcmp(name, 'PixelData') % if PixelData in SQ/data was caught
            iPixelData = iPre*2-1; % start of PixelData tag in bytes
            break; % done
        end
        if allRead
            info = ['End of file reached: likely error: ' s.Filename];  
            break; % give up
        else % in case PixelData in SQ was caught
            b = [b fread(fid, inf, '*uint16')']; %#ok read all
            len = numel(b)-6; % update length
            i = iPre; % re-do the previous item
            allRead = true;
        end
    end
    iPre = i; % backup it, also useful for PixelData
    
    [dat, name, info, i, tg] = read_item(i);
    if ~fullHdr && tg>dict.tag(end), break; end % done for partial hdr
    if strncmp(info, 'Given up', 8), break; end
    if isnumeric(name) || isempty(dat), continue; end
    if strcmp(name, 'Manufacturer') && ~isempty(dict.vendor) ...
            && ~strncmp(dict.vendor, dat, 2)
        dict_full = dicm_dict(dat); % update vendor
        if fullHdr, dict = [];
        else dict = dicm_dict(dat, dict.name);
        end
        [s, info] = dicm_hdr(fname, dict, iFrames);
        return;
    elseif strcmp(name, 'TransferSyntaxUID')
        expl = ~strcmp(dat, '1.2.840.10008.1.2'); % may be wrong for some
    end
    s.(name) = dat;
end

i = (iPixelData+1) / 2; % start of PixelData tag in uint16
if isTruncated
    iPixelData = iPixelData +   7; i=i+2;
elseif explVR
    iPixelData = iPixelData + 143; i=i+4; % extra vr(2) + pad(2)
else
    iPixelData = iPixelData + 139; i=i+2;
end
s.PixelData.Start = uint32(iPixelData);
s.PixelData.Bytes = typecast(b(i+(0:1)), 'uint32');

% if iPixelData is not right, re-do with full header
if ~fullHdr
    fseek(fid, 0, 1); % end of file
    if ftell(fid)-s.PixelData.Start-s.PixelData.Bytes > 15 % ==0 is too strict
        [s, info] = dicm_hdr(fname, [], iFrames); % full hdr
        return;
    end
end

if isfield(s, 'CSAImageHeaderInfo') % Siemens CSA image header (slow)
    s.CSAImageHeaderInfo = read_csa(s.CSAImageHeaderInfo);
end
if isfield(s, 'CSASeriesHeaderInfo') % series header
    s.CSASeriesHeaderInfo = read_csa(s.CSASeriesHeaderInfo);
end
if isfield(s, 'ProtocolDataBlock') % GE
    s.ProtocolDataBlock = read_ProtocolDataBlock(s.ProtocolDataBlock);
end
return;

% Nested function: read dicom item. Called by dicm_hdr and read_sq
function [dat, name, info, i, tag] = read_item(i)
persistent len16 chDat;
if isempty(len16)
    len16 = 'AE AS AT CS DA DS DT FD FL IS LO LT PN SH SL SS ST TM UI UL US';
    chDat = 'AE AS CS DA DS DT IS LO LT PN SH ST TM UI UT';
end
dat = []; name = nan; info = ''; 
vr = 'CS'; % CS for Manufacturer and TransferSyntaxUID

group = b(i); i=i+1;
elmnt = b(i); i=i+1;
tag = uint32(group)*65536 + uint32(elmnt);
if tag == 4294893581 % FFFE E00D ItemDelimitationItem for Philips SQ
    i = i+2; % skip length, in case there is another SQ Item
    name = '';
    return;
end

explVR = expl || group==2;
if explVR, vr = char(typecast(b(i), 'uint8')); i=i+1; end % 2-byte VR

if ~explVR % implicit, length irrevalent to VR
    n = typecast(b(i+(0:1)), 'uint32'); i=i+2;
elseif ~isempty(strfind(len16, vr)) % data length in uint16
    n = b(i); i=i+1;
else % length in uint32: skip 2 bytes
    n = typecast(b(i+(1:2)), 'uint32'); i=i+3;
end
if n<1, return; end % empty val

% Look up item name in dictionary
n = double(n)/2;
ind = find(dict.tag == tag);
if ~isempty(ind)
    name = dict.name{ind};
    if ~explVR || (strcmp(vr, 'UN') || strcmp(vr, 'OB'))
        vr = dict.vr{ind};
    end
elseif tag==524400 % in case not in dict
    name = 'Manufacturer';
elseif tag==131088 % can't skip TransferSyntaxUID even if not in dict
    name = 'TransferSyntaxUID';
elseif fullHdr
    if elmnt==0, i=i+n; return; end % skip GroupLength
    name = sprintf('Private_%04x_%04x', group, elmnt);
    if ~explVR, vr = 'UN'; end
elseif n<2147483647.5 % no skip for SQ with length 0xffffffff
    i=i+n; return;
end
% compressed PixelData can be 0xffffffff
if ~explVR && n==2147483647.5, vr = 'SQ'; end % best guess
if (n+i>len) && (~strcmp(vr, 'SQ')), i = i+n; return; end % re-do
% fprintf('(%04x %04x) %s %s\n', group, elmnt, vr, name);

% Decode data length and type of an item by VR
if ~isempty(strfind(chDat, vr)) % char data
    dat = deblank(char(typecast(b(i+(0:n-1)), 'uint8'))); i=i+n;
    if strcmp(vr, 'DS') || strcmp(vr, 'IS')
        dat = sscanf(dat, '%f%*c'); % like 1\2\3
    end
elseif strcmp(vr, 'SQ')
    isPerFrameSQ = strcmp(name, 'PerFrameFunctionalGroupsSequence');
    [dat, info, i] = read_sq(i, min(i+n,len), isPerFrameSQ);
    return;
else % numeric data, or UN
    fmt = vr2format(vr);
    if isempty(fmt)
        info = sprintf('Given up: Invalid VR (%d %d) for %s', vr, name);
        fprintf(2, ' %s\n', info);
    else
        dat = typecast(b(i+(0:n-1)), fmt)'; i=i+n;
    end
end
end % nested func

% Nested function: decode SQ, called by read_item (recursively)
function [rst, info, i] = read_sq(i, nEnd, isPerFrameSQ)
rst = []; info = ''; j = 0;

while i<nEnd
    tag = typecast(b(i+(0:1)), 'uint32'); i=i+2;
    n = typecast(b(i+(0:1)), 'uint32'); i=i+2; % n may be 0xffff ffff
    if tag ~= 3758161918, return; end % only do FFFE E000, Item

    if isPerFrameSQ && ~ischar(iFrames)
        if j==0, i0 = i; j = 1; % always read 1st frame
        elseif j==1 % always read 2nd frame, and find ind for all frames
            j = 2; iItem = 2;
            tag1 = typecast(tag1, 'uint8');
            tag1 = tag1([3 4 1 2]);
            ind = strfind(typecast(b(i0:(iPixelData+1)/2), 'uint8'), tag1);
            ind = (ind-1)/2 + i0;
            iFrames = unique([1 2 round(iFrames) numel(ind)]);
        else
            iItem = iItem + 1;
            j = iFrames(iItem);
            i = ind(j);
        end
    else
        j = j + 1;
    end
    
    Item_n = sprintf('Item_%g', j);
    n = min(i+double(n)/2, nEnd);
    
    while i<n
        [dat1, name1, info, i, tag] = read_item(i);
        if isnumeric(name1), continue; end % 0-length or skipped item
        if isempty(dat1), break; end
        if isempty(rst), tag1 = tag; end % first wanted tag in SQ
        rst.(Item_n).(name1) = dat1;
    end
end
end % nested func
end % main func

% subfunction: return format str for typecast according to VR
function fmt = vr2format(vr)
switch vr 
    case 'OB', fmt = 'uint8';
    case 'UN', fmt = 'uint8';
    case 'AT', fmt = 'uint16';
    case 'OW', fmt = 'uint16';
    case 'US', fmt = 'uint16';
    case 'SS', fmt = 'int16'; 
    case 'UL', fmt = 'uint32';
    case 'SL', fmt = 'int32';
    case 'FL', fmt = 'single'; 
    case 'FD', fmt = 'double';
    otherwise, fmt = '';
end
end

% subfunction: decode Siemens CSA image and series header
function csa = read_csa(csa)
b = csa';
if ~strcmp(char(b(1:4)), 'SV10'), return; end % no op if not SV10
chDat = 'AE AS CS DA DT LO LT PN SH ST TM UI UN UT';
i = 8; % 'SV10' 4 3 2 1
try %#ok in case of error, we return the original uint8
    nField = typecast(b(i+(1:4)), 'uint32'); i=i+8;
    for j = 1:nField
        i=i+68; % name(64) and vm(4)
        vr = char(b(i+(1:2))); i=i+8; % vr(4), syngodt(4)
        n = typecast(b(i+(1:4)), 'int32'); i=i+8;
        if n<1, continue; end % skip name decoding, faster
        ind = find(b(i-84+(1:64))==0, 1) - 1;
        name = char(b(i-84+(1:ind)));
        % fprintf('%s %3g %s\n', vr, n, name);

        dat = [];
        for k = 1:n % n is often 6, but often only the first contains value
            len = typecast(b(i+(1:4)), 'int32'); i=i+16;
            if len<1, i = i+double(n-k)*16; break; end % rest are empty too
            foo = char(b(i+(1:len)));
            i = i + ceil(double(len)/4)*4; % multiple 4-byte
            if isempty(strfind(chDat, vr))
                dat(end+1,1) = str2double(foo); %#ok numeric to double
            else
                dat = deblank(foo);
                i = i+double(n-1)*16;
                break; % char parameters always have 1 item only
            end
        end
        if ~isempty(dat), rst.(name) = dat; end
    end
    csa = rst;
end
end

% subfunction: decode GE ProtocolDataBlock
function ch = read_ProtocolDataBlock(ch)
n = typecast(ch(1:4), 'int32') + 4; % nBytes, zeros may be padded to make 4x
if ~all(ch(5:7) == [31 139 8]'), return; end % gz signature
tmp = tempname; % temp gz file
fid = fopen([tmp '.gz'], 'w');
if fid<0, return; end
fwrite(fid, ch(5:n), 'uint8');
fclose(fid);
cln = onCleanup(@() delete([tmp '.*'])); % delete gz and unziped files

try
    gunzip([tmp '.gz']);
    fid = fopen(tmp);
    b = fread(fid, '*char')';
    fclose(fid);
catch
    return;
end

try %#ok
    i = 1; n = numel(b);
    while i<n
        nam = strtok(b(i:n), ' "'); i = i + numel(nam) + 2; % VIEWORDER "1"
        val = strtok(b(i:n),  '"'); i = i + numel(val) + 2;
        if strcmp(val(end), ';'), val(end) = []; end
        foo = str2num(val); %#ok take care of multiple numbers
        if ~isempty(foo), val = foo; end % convert into num if possible
        rst.(nam) = val;
    end
    ch = rst;
end
end

%% subfunction: read PAR file, return struct like that from dicm_hdr.
function [s, err] = philips_par(fname)
err = '';
if numel(fname)>4 && strcmpi(fname(end+(-3:0)), '.REC')
    fname(end+(-3:0)) = '.PAR';
    if ~exist(fname, 'file'), fname(end+(-3:0)) = '.par'; end
end
fid = fopen(fname);
if fid<0, s = []; err = ['File not exist: ' fname]; return; end
str = fread(fid, inf, '*char')'; % read all as char
fname = fopen(fid); % name with full path
fclose(fid);

% In V4, offcentre and Angulation labeled as y z x, but actually x y z. We
% try not to use these info
key = 'image export tool';
i = strfind(lower(str), key) + numel(key);
if isempty(i), err = 'Not PAR file'; s = []; return; end
C = textscan(str(i:end), '%s', 1);
s.SoftwareVersion = C{1}{1};
if strncmpi(s.SoftwareVersion, 'V3', 2)
    err = 'V3 PAR file is not supported';
    fprintf(2, ' %s. \n', err);
    s = []; return;
end

s.IsPhilipsPAR = true;
s.PatientName = par_key('Patient name', '%c');
s.StudyDescription = par_key('Examination name', '%c');
[pth, nam] = fileparts(fname);
s.SeriesDescription = nam;
s.ProtocolName = par_key('Protocol name', '%c');
foo = par_key('Examination date/time', '%s');
foo = foo(isstrprop(foo, 'digit'));
s.AcquisitionDateTime = foo;
% s.SeriesType = strkey(str, 'Series Type', '%c');
s.SeriesNumber = par_key('Acquisition nr');
s.InstanceNumber = 1; % make dicm2nii.m happy
% s.SamplesPerPixel = 1;
% s.ReconstructionNumberMR = strkey(str, 'Reconstruction nr', '%g');
% s.MRSeriesScanDuration = strkey(str, 'Scan Duration', '%g');
s.NumberOfEchoes = par_key('Max. number of echoes');
nSL = par_key('Max. number of slices/locations');
s.SlicesPerVolume = nSL;
s.NumberOfTemporalPositions = par_key('Max. number of dynamics');
foo = par_key('Patient position', '%c');
if isempty(foo), foo = par_key('Patient Position', '%c'); end
if ~isempty(foo)
    if numel(foo)>4, s.PatientPosition = foo(regexp(foo, '\<.')); 
    else s.PatientPosition = foo; 
    end
end
foo = par_key('Preparation direction', '%s');
if ~isempty(foo)
    s.Stack.Item_1.MRStackPreparationDirection = foo(regexp(foo, '\<.'));
end
s.ScanningSequence = par_key('Technique', '%s'); % ScanningTechnique
s.ImageType = s.ScanningSequence;
% foo = strkey(str, 'Scan resolution', '%g'); % before reconstruction
% s.AcquisitionMatrix = [foo(1) 0 0 foo(2)]'; % depend on slice ori
s.RepetitionTime = par_key('Repetition time');
% FOV = par_key('FOV'); % (ap,fh,rl) [mm] 
% FOV = FOV([3 1 2]); % x y z
s.WaterFatShift = par_key('Water Fat shift');
rotAngle = par_key('Angulation midslice'); % (ap,fh,rl) deg
rotAngle = rotAngle([3 1 2]);
posMid = par_key('Off Centre midslice'); % (ap,fh,rl) [mm]
s.Stack.Item_1.MRStackOffcentreAP = posMid(1);
s.Stack.Item_1.MRStackOffcentreFH = posMid(2);
s.Stack.Item_1.MRStackOffcentreRL = posMid(3);
posMid = posMid([3 1 2]); % better precision than those in the table
if par_key('MTC') % motion correction?
    s.ImageType = [s.ImageType '\MOCO\'];
end
s.EPIFactor = par_key('EPI factor');
% s.DynamicSeries = strkey(str, 'Dynamic scan', '%g'); % 0 or 1
isDTI = par_key('Diffusion')>0;
if isDTI
    s.ImageType = [s.ImageType '\DIFFUSION\'];
    s.DiffusionEchoTime = par_key('Diffusion echo time'); % ms
end

% Get list of para meaning for the table, and col index of each para
i1 = strfind(str, '= IMAGE INFORMATION DEFINITION ='); i1 = i1(end);
ind = strfind(str(i1:end), [char([13 10]) '#']) + i1; % start of # lines
for i = 1:9 % find the empty line before column descrip
    [~, foo] = strtok(str(ind(i):ind(i+1)-3)); % -3 remove # and [13 10]
    if isempty(foo), break; end 
end
j = 1; 
for i = i+1:numel(ind)
    [~, foo] = strtok(str(ind(i):ind(i+1)-3)); % -3 remove # and [13 10]
    if isempty(foo), break; end % the end of the col label
    foo = strtrim(foo);
    i3 = strfind(foo, '<');
    i2 = strfind(foo, '(');
    if isempty(i3), i3 = i2(1); end
    colLabel{j} = strtrim(foo(1:i3(1)-1)); %#ok para name
    nCol = sscanf(foo(i2(end)+1:end), '%g');
    if isempty(nCol), nCol = 1; end
    iColumn(j) = nCol; %#ok number of columns in the table for this para
    j = j + 1;
end
iColumn = cumsum([1 iColumn]); % col start ind for corresponding colLabel
keyInLabel = @(key)strcmp(colLabel, key);
colIndex = @(key)iColumn(keyInLabel(key));

i1 = strfind(str, '= IMAGE INFORMATION ='); i1 = i1(end);
ind = strfind(str(i1:end), char([13 10])) + i1 + 1; % start of a line
for i = 1:9
    foo = sscanf(str(ind(i):end), '%g', 1);
    if ~isempty(foo), break; end % get the first number
end
while str(ind(i))==13, i = i+1; end % skip empty lines (only one)
str = str(ind(i):end); % now start of the table

i1 = strfind(str, char(10));
para = sscanf(str(1:i1(1)), '%g'); % 1st row
n = numel(para); % number of items each row, 41 for V4
para = sscanf(str, '%g'); % read all numbers
nImg = floor(numel(para) / n); 
para = reshape(para(1:n*nImg), n, nImg)'; % whole table now
s.NumberOfFrames = nImg;

s.Dim3IsVolume = (diff(para(1:2, colIndex('slice number'))) == 0);
if s.Dim3IsVolume, iVol = 1:(nImg/nSL);
else iVol = 1:nSL:nImg;
end

imgType = para(iVol, colIndex('image_type_mr')); % 0 mag; 3, phase?
if any(diff(imgType) ~= 0) % more than 1 type of image
    s.ComplexImageComponent = 'MIXED';
    s.VolumeIsPhase = (imgType==3); % one for each vol
    s.LastFile.RescaleIntercept = para(end, colIndex('rescale intercept'));
    s.LastFile.RescaleSlope = para(end, colIndex('rescale slope'));
elseif imgType(1)==0, s.ComplexImageComponent = 'MAGNITUDE';
elseif imgType(1)==3, s.ComplexImageComponent = 'PHASE';
end

% These columns should be the same for all images: 
cols = {'image pixel size' 'recon resolution' 'image angulation' ...
    'slice thickness' 'slice gap' 'slice orientation' 'pixel spacing'};
if ~strcmp(s.ComplexImageComponent, 'MIXED')
    cols = [cols {'rescale intercept' 'rescale slope'}];
end
ind = [];
for i = 1:numel(cols)
    j = find(keyInLabel(cols{i}));
    if isempty(j), continue; end
    ind = [ind iColumn(j):iColumn(j+1)-1]; %#ok
end
foo = para(:, ind);
foo = abs(diff(foo));
if any(foo(:) > 1e-5)
    err = sprintf('Inconsistent image size, bits etc: %s', fname);
    fprintf(2, ' %s. \n', err);
    s = []; return;
end

% getTableVal('echo number', 'EchoNumber', 1:nImg);
% getTableVal('dynamic scan number', 'TemporalPositionIdentifier', 1:nImg);
getTableVal('image pixel size', 'BitsAllocated');
getTableVal('recon resolution', 'Columns');
s.Rows = s.Columns(2); s.Columns = s.Columns(1);
getTableVal('rescale intercept', 'RescaleIntercept');
getTableVal('rescale slope', 'RescaleSlope');
getTableVal('window center', 'WindowCenter', 1:nImg);
getTableVal('window width', 'WindowWidth', 1:nImg);
mx = max(s.WindowCenter + s.WindowWidth/2);
mn = min(s.WindowCenter - s.WindowWidth/2);
s.WindowCenter = round((mx+mn)/2);
s.WindowWidth = ceil(mx-mn);
getTableVal('slice thickness', 'SliceThickness');
getTableVal('echo_time', 'EchoTime');
% getTableVal('dyn_scan_begin_time', 'TimeOfAcquisition', 1:nImg);
if isDTI
    getTableVal('diffusion_b_factor', 'B_value', iVol);
    fld = 'DiffusionGradientDirection';
    getTableVal('diffusion', fld, iVol);
    if isfield(s, fld), s.(fld) = s.(fld)(:, [3 1 2]); end
end
getTableVal('TURBO factor', 'TurboFactor');

% Rotation order and signs are figured out by try and err, not 100% sure
ca = cosd(rotAngle); sa = sind(rotAngle);
rx = [1 0 0; 0 ca(1) -sa(1); 0 sa(1) ca(1)]; % standard 3D rotation
ry = [ca(2) 0 sa(2); 0 1 0; -sa(2) 0 ca(2)];
rz = [ca(3) -sa(3) 0; sa(3) ca(3) 0; 0 0 1];
R = rx * ry * rz; % seems right for Philips, but standard seems rz*ry*rx

getTableVal('slice orientation', 'SliceOrientation'); % 1/2/3 for TRA/SAG/COR
iOri = mod(s.SliceOrientation+1, 3) + 1; % [1 2 3] to [3 1 2]
if iOri == 1 % Sag
    s.SliceOrientation = 'SAGITTAL';
    ixyz = [2 3 1];
    R(:,[1 3]) = -R(:,[1 3]); % change col sign according to iOri
elseif iOri == 2 % Cor
    s.SliceOrientation = 'CORONAL';
    ixyz = [1 3 2];
    R(:,3) = -R(:,3);
else % Tra
    s.SliceOrientation = 'TRANSVERSAL';
    ixyz = [1 2 3];
end
% bad precision for some PAR, 'pixel spacing' and 'slice gap', but it is wrong
% to use FOV, maybe due to partial Fourier?
getTableVal('pixel spacing', 'PixelSpacing');
s.PixelSpacing = s.PixelSpacing(:);
getTableVal('slice gap', 'SpacingBetweenSlices');
s.SpacingBetweenSlices = s.SpacingBetweenSlices + s.SliceThickness;
% s.PixelSpacing = FOV(ixyz(1:2)) ./ [s.Columns s.Rows]';
% s.SpacingBetweenSlices = FOV(ixyz(3)) ./ nSL;

% iPhase = strfind('LRAPSIFH', s.Stack.Item_1.MRStackPreparationDirection(1));
% iPhase = ceil(iPhase/2); 
% if iPhase>3, iPhase = 3; end % 1/2/3 for LR AP FH
% foo = 'COL';
% if iPhase == ixyz(1), foo = 'ROW'; end
% s.InPlanePhaseEncodingDirection = foo;

R = R(:, ixyz); % dicom rotation matrix
s.ImageOrientationPatient = R(1:6)';
R = R * diag([s.PixelSpacing; s.SpacingBetweenSlices]);
R = [R posMid; 0 0 0 1]; % 4th col is mid slice center position
% x = ([s.Columns s.Rows nSL]' -1) / 2; % some V4.2 seeem to use this
x = [s.Columns s.Rows nSL-1]' / 2; % ijk of mid slice center 
R = R / [eye(3) x; 0 0 0 1]; % dicom xform matrix
y = R * [0 0 nSL-1 1]'; % last slice position
if sign(y(iOri)-R(iOri,4)) == sign(y(iOri)-posMid(iOri))
    s.ImagePositionPatient = R(1:3,4); % slice direction in R is correct
    s.LastFile.ImagePositionPatient = y(1:3);
else
    s.ImagePositionPatient = y(1:3); % slice direction in R is opposite
    s.LastFile.ImagePositionPatient = R(1:3,4);
end
s.Manufacturer = 'Philips';
s.Filename = fullfile(pth, [nam '.REC']); % for dicm_img
s.PixelData.Start = 0; % for dicm_img.m
s.PixelData.Bytes = s.Rows * s.Columns * nImg * s.BitsAllocated / 8;

    % nested function: set field if the key is in colTable
    function getTableVal(key, fldname, iRow)
    if nargin<3, iRow = 1; end
    iCol = find(keyInLabel(key));
    if isempty(iCol), return; end
    s.(fldname) = para(iRow, iColumn(iCol):iColumn(iCol+1)-1);
    end

    % nested subfunction: return value specified by key in PAR file
    function val = par_key(key, fmt)
    if nargin<2 || isempty(fmt), fmt = '%g';  end
    i1 = regexp(str, ['\n.\s{1,}' key '\s{0,}[(<\[:]']);
    if isempty(i1)
        if strcmp(fmt, '%g'), val = [];
        else val = '';
        end
        return; 
    end
    i1 = i1(1) + 1; % skip '\n'
    i2 = find(str(i1:end)==char(10), 1, 'first') + i1 - 2;
    ln = str(i1:i2); % the line
    i1 = strfind(ln, ':') + 1;
    val = sscanf(ln(i1(1):end), fmt); % convert based on fmt, re-use fmt
    if isnumeric(val), val = double(val);
    else val = strtrim(val);
    end
    end
end

%% subfunction: read AFNI HEAD file, return struct like that from dicm_hdr.
function [s, err] = afni_head(fname)
persistent SN;
if isempty(SN), SN = 1; end
err = '';
if numel(fname)>5 && strcmp(fname(end+(-4:0)), '.BRIK')
    fname(end+(-4:0)) = '.HEAD';
end
fid = fopen(fname);
if fid<0, s = []; err = ['File not exist: ' fname]; return; end
str = fread(fid, inf, '*char')';
fname = fopen(fid);
fclose(fid);

i = strfind(str, 'DATASET_DIMENSIONS');
if isempty(i), s = []; err = 'Not brik header file'; return; end

% these make dicm_nii.m happy
[~, foo] = fileparts(fname);
s.IsAFNIHEAD = true;
s.ProtocolName = foo;
s.SeriesNumber = SN; SN = SN+1; % make it unique for multilple files
s.InstanceNumber = 1;
s.ImageType = afni_key('TYPESTRING');

foo = afni_key('BYTEORDER_STRING');
if strcmp(foo(1), 'M'), err = 'BYTEORDER_STRING not supported'; s = []; return; end

foo = afni_key('BRICK_FLOAT_FACS');
if any(diff(foo)~=0), err = 'Inconsistent BRICK_FLOAT_FACS'; 
    s = []; return; 
end
if foo(1)==0, foo = 1; end
s.RescaleSlope = foo(1);
s.RescaleIntercept = 0;

foo = afni_key('BRICK_TYPES');
if any(diff(foo)~=0), err = 'Inconsistent DataType'; s = []; return; end
foo = foo(1);
if foo == 0
    s.BitsAllocated =  8; s.PixelData.Format = '*uint8';
elseif foo == 1
    s.BitsAllocated = 16; s.PixelData.Format = '*int16';
elseif foo == 3
    s.BitsAllocated = 32; s.PixelData.Format = '*single';
else
    error('Unsupported BRICK_TYPES: %g', foo);
end

hist = afni_key('HISTORY_NOTE');
i = strfind(hist, 'Time:') + 6;
if ~isempty(i)
    dat = sscanf(hist(i:end), '%11c', 1); % Mar  1 2010
    dat = datenum(dat, 'mmm dd yyyy');
    s.AcquisitionDateTime = datestr(dat, 'yyyymmdd');
end
i = strfind(hist, 'Sequence:') + 9;
if ~isempty(i), s.ScanningSequence = strtok(hist(i:end), ' '); end
i = strfind(hist, 'Studyid:') + 8;
if ~isempty(i), s.StudyID = strtok(hist(i:end), ' '); end
% i = strfind(hist, 'Dimensions:') + 11;
% if ~isempty(i)
%     dimStr = strtok(hist(i:end), ' ') % 64x64x35x92
% end
% i = strfind(hist, 'Orientation:') + 12;
% if ~isempty(i)
%     oriStr = strtok(hist(i:end), ' ') % LAI
% end
i = strfind(hist, 'TE:') + 3;
if ~isempty(i), s.EchoTime = sscanf(hist(i:end), '%g', 1) * 1000; end

% foo = afni_key('TEMPLATE_SPACE'); % ORIG/TLRC
% INT_CMAP
foo = afni_key('SCENE_DATA');
s.TemplateSpace = foo(1)+1; %[0] 0=+orig, 1=+acpc, 2=+tlrc
if foo(2)==9, s.ImageType =[s.ImageType '\DIFFUSION\']; end
% ori = afni_key('ORIENT_SPECIFIC')+1;
% orients = [1 -1 -2 2 3 -3]; % RL LR PA AP IS SI
% ori = orients(ori) % in dicom/afni LPS, 
% seems always [1 2 3], meaning AFNI re-oriented the volome

% no read/phase/slice dim info, so following 3D info are meaningless
dim = afni_key('DATASET_DIMENSIONS');
s.Columns = dim(1); s.Rows = dim(2); s.SlicesPerVolume = dim(3);
R = afni_key('IJK_TO_DICOM_REAL'); % IJK_TO_DICOM is always straight?
if isempty(R), R = afni_key('IJK_TO_DICOM'); end
R = reshape(R, [4 3])';
s.ImagePositionPatient = R(:,4); % afni_key('ORIGIN') can be wrong
y = [R; 0 0 0 1] * [0 0 dim(3)-1 1]';
s.LastFile.ImagePositionPatient = y(1:3);
R = R(1:3, 1:3);
R = R ./ (ones(3,1) * sqrt(sum(R.^2)));
s.ImageOrientationPatient = R(1:6)';
foo = afni_key('DELTA');
s.PixelSpacing = foo(1:2);
% s.SpacingBetweenSlices = foo(3);
s.SliceThickness = foo(3);
foo = afni_key('BRICK_STATS');
foo = reshape(foo, [2 numel(foo)/2]);
mn = min(foo(1,:)); mx = max(foo(2,:));
s.WindowCenter = (mx+mn)/2;
s.WindowWidth = mx-mn;
foo = afni_key('TAXIS_FLOATS'); %[0]:0; 
if ~isempty(foo), s.RepetitionTime = foo(2)*1000; end

foo = afni_key('TAXIS_NUMS'); % [0]:nvals; [1]: 0 or nSL normally
if ~isempty(foo)
    inMS = foo(3)==77001;
    foo = afni_key('TAXIS_OFFSETS');
    if inMS, foo = foo/1000; end
    if ~isempty(foo), s.MosaicRefAcqTimes = foo; end
end

foo = afni_key('DATASET_RANK'); % [3 nvals]
dim(4) = foo(2);
s.NumberOfTemporalPositions = dim(4);
% s.NumberOfFrames = dim(4)*dim(3);
 
s.Manufacturer = '';
s.Filename = strrep(fname, '.HEAD', '.BRIK');
s.PixelData.Start = 0; % make it work for dicm_img.m
s.PixelData.Bytes = prod(dim(1:4)) * s.BitsAllocated / 8;

    % subfunction: return value specified by key in afni header str
    function val = afni_key(key)
    i1 = regexp(str, ['\nname\s{0,}=\s{0,}' key '\n']); % line 'name = key'
    if isempty(i1), val = []; return; end
    i1 = i1(1) + 1;
    i2 = regexp(str(1:i1), 'type\s{0,}=\s{0,}\w*-attribute\n');
    keyType = sscanf(str(i2(end):i1), 'type%*c=%*c%s', 1); %'string-attribute'
    i1 = find(str(i1:end)==char(10), 1, 'first') + i1;
    count = sscanf(str(i1:end), 'count%*c=%*c%g', 1);
    if strcmp(keyType, 'string-attribute')
        i1 = find(str(i1:end)=='''', 1, 'first') + i1;
        val = str(i1+(0:count-2));
    else
        i1 = find(str(i1:end)==char(10), 1, 'first') + i1;
        val = sscanf(str(i1:end), '%g', count);
    end
    end
end
