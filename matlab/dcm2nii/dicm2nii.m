function varargout = dicm2nii(src, dataFolder, varargin)
% DICM2NII converts dicom files into NIfTI or img/hdr files. 
% 
% DICM2NII(dcmSource, niiFolder, outFormat, MoCoOption, subjName)
% 
% The input arguments are all optional:
%  1. source file or folder. It can be a zip or tgz file, a folder containing
%     dicom files, or other convertible files. It can also contain wildcards
%     like 'run1_*' for all files start with 'run1_'.
%  2. folder to save result files.
%  3. output file format:
%      0 or 'nii'    for uncompressed single file.
%      1 or 'nii.gz' for compressed single file (default).
%      2 or 'img'    for triplet files (img/hdr/mat) without compression.
%      3 or 'img.gz' for compressed triplet files (img/hdr/mat with gz).
%  4. MoCo series options:
%      0 create files for both original and MoCo series.
%      1 ignore MoCo series if both present (default).
%      2 ignore original series if both present.
%     Note that if only one of the two series is present, it will be created
%     always.
%  5. subject name the data is from. The code can do only one subject each time.
%     If you have a folder or zip/tgz file for multiple subjects (not
%     recommended), you can specify which subject to convert with the 5th input.
%     The name should be the Patient Name entered at scanner console for in
%     format of LastName^FirstName, such as Smith^John, or simply the Last name
%     if no First name was provided to the console. If you already have mixed
%     subject data in a folder, you can let the code return the unconverted
%     subject names in the second output argument, and then provide each of
%     subjects as the 5th input to convert data into specific subject folder
%     (see example below).
% 
% The optional output are converted (1st) and unconverted (2nd) PatientName.
% 
% Typical examples:
%  dicm2nii; % bring up user interface if there is no input argument
%  dicm2nii('D:/myProj/zip/subj1.zip', 'D:/myProj/subj1/data'); % zip file
%  dicm2nii('D:/myProj/subj1/dicom/', 'D:/myProj/subj1/data'); % folder
% 
% Less useful examples:
%  dicm2nii('D:/myProj/dicom/', 'D:/myProj/subj2/data', [], [], 'subj2');
%  dicm2nii('D:/myProj/dicom/run2*', 'D:/myProj/subj/data');
% 
% Example to deal with multiple subjects:
%  [converted, otherSubj] = dicm2nii('D:/myProj/dicom/', 'D:/myProj/subj1/data');
%  movefile('D:/myProj/subj1', ['D:/myProj/' converted]); % rename folder
%  In case of multiple subjects, above command will convert one of the subjects,
%  and return the unconverted subjects in second output. You can convert other
%  subjects one by one using script like this:
%     for i = 1:length(otherSubj)
%         dataDir = ['D:/myProj/' otherSubj{i} '/data'];
%         dicm2nii('D:/myProj/dicom/', dataDir, [], [], otherSubj{i});
%     end
% 
% If there is no input, or any of the first two inputs is empty, a graphic user
% interface will appear.
% 
% If the first input is a zip/tgz file, such as those downloaded from dicom
% server, DICM2NII will extract files into a temp folder, create NIfTI files
% into the data folder, and then delete the temp folder. For this reason, it is
% better to keep the compressed file as backup.
% 
% If a folder is the data source, DICM2NII will convert all files in the folder
% and its subfolders (you don't need to sort files for different series).
% 
% Please note that, if a file in the middle of a series is missing, the series
% will be skipped without converting, and a warning message will be shown as red
% text in Command Windows, and saved into a text file the data folder.
% 
% Slice timing information, if available, is stored in nii header, such as
% slice_code and slice_duration. But the simple way may be to use the field
% SliceTiming in dcmHeaders.mat. That timing is actually those numbers for FSL
% when using custom slice timing. This is the universal method to specify any
% kind of slice order, and for now, is the only way which works for multiband.
% Slice order is one of the most confusing parameters, and it is recommended to
% use this method to avoid mistake. To convert this timing into slice order for
% SPM: [~, spm_order] = sort(s.SliceTiming, 'descend');
% 
% If there is DTI series, bval and bvec files will be generated for FSL etc. A
% Matlab data file, dcmHeaders.mat, is always saved into the data folder. This
% file contains dicom header from the first file for created series and some
% information from last file in field LastFile. For DTI series, B_value and
% DiffusionGradientDirection for all directions are saved into the mat file. For
% MoCo series, motion parameters, RBMoCoTrans and RBMoCoRot, are also saved.
% 
% Some information, such as TE, phase encoding direction and effective dwell
% time are stored in descrip of nii header. These are useful for fieldmap B0
% unwarp correction. Acquisition start time and date are also stored, and this
% may be useful if one wants to align the functional data to some physiological
% recording, like pulse, respiration or ECG.
% 
% The output file names adopt SeriesDescription or ProtocolName of each series
% used on scanner console. If both original and MoCo series are created, '_MoCo'
% will be appended for MoCo series. For phase image, such as those from field
% map, '_phase' will be appended to the name. In case of name conflict,
% SeriesNumber, such as '_s005', will be appended to make file names unique. It
% is suggested to use short and descriptive SeriesDescription on the scanner
% console, and use names containing only letters, numbers and underscores.
% 
% Some of the parameters may not be available for all vendors. For example,
% there is no slice order information in Philips data. Please report any bug to
% xiangrui.li@gmail.com or at
% http://www.mathworks.com/matlabcentral/fileexchange/42997-dicom-to-nifti-converter.

% Thanks to:
% Jimmy Shen's Tools for NIfTI and ANALYZE image,
% Chris Rorden's dcm2nii pascal source code,
% Przemyslaw Baranski for direction cosine matrix to quaternions. 

% History (yymmdd):
% 130512 Publish to CCBBI users (Xiangrui Li).
% 130513 Convert img from uint16 to int16 if range allows;
%        Expand output file format to img/hdr/mat.
% 130515 Change creation order to acquisition order (more natural).
%        If MoCo series is included, append _MoCo in file names.
% 130516 Use SpacingBetweenSlices, if exists, for SliceThickness. 
% 130518 Use NumberOfImagesInMosaic in CSA header (work for some old data).
% 130604 Add scl_inter/scl_slope and special naming for fieldmap.
% 130614 Work out the way to get EffectiveEchoSpacing for B0 unwarp.
% 130616 Add needed dicom field check, so it won't err later.
% 130618 Reorient if non-mosaic or slice_dim is still 3 and no slice flip.
% 130619 Simplify DERIVED series detection. No '_mag' in fieldmap name.
% 130629 Improve the method to get phase direction;
%        Permute img dim1&2 (no -90 rotation) & simplify xform accordingly.
% 130711 Make MoCoOption smarter: create nii if only 1 of 2 series exists.
% 130712 Remove 5th input (allHeader). Save memory by using partial header.
% 130712 Bug fix: dim_info with reorient. No problem since no EPI reorient.
% 130715 Use 2 slices for xform. No slice flip needed except revNum mosaic.
% 130716 Take care of lower/upper cases for output file names;
%        Apply scl_slope and inter to img if range allows and no rounding;
%        Save motion parameters, if any, into dcmHeader.mat.
% 130722 Ugly fix for isMos, so it works for '2004A 4VA25A' phase data;
%        Store dTE instead of TE if two TE are used, such as fieldmap.
% 130724 Add two more ways for dwell time, useful for '2004A 4VA25A' dicom.
% 130801 Can't use DERIVED since MoCoSeries may be labeled as DERIVED.
% 130807 Check PixelSpacing consistency for a series;
%        Prepare to publish to Matlab Central.
% 130809 Add 5th input for subjName, so one can choose a subject for nii.
% 130813 Store ImageComments, if exists and is meaningful, into aux_file.
% 130818 Expand source to dicom file(s) and wildcards like run1*.dcm.
%        Update fields in dcmHeader.mat, rather than overwriting the file.
%        Include save_nii etc in the code for easy distribution.
% 130821 Bug fix for cellstr input as dicom source.
%        Change file name from dcm2nii.m to reduce confusion from MRICron.
%        GUI implemented into the single file.
% 130823 Remove dependency on Image Processing Toolbox.
% 130826 Bug fix for '*' src input. Minor improvement for dicm_hdr.
% 130827 Try and suggest to use pigz for compression (thanks Chris R.).
% 130905 Avoid the missing-field error for DTI data with 2 excitations.
%        Protect GUI from command line plotting.
% 130912 Use lDelayInTR for slice_dur, possibly useful for old data.
% 130916 Store B_matrix for DTI image, if exists.
% 130919 Make the code work for GE and Philips dicom at Chris R website.
% 130922 Remove dependence on normc from nnet toolbox (thank Zhiwei);
%        Prove no slice order info in Philips, at least for Intera 10.4.1.
% 130923 Make the code work for Philips PAR/REC pair files.
% 130926 Take care of non-mosaic DTI for Siemens (img/bval/bvec);
% 130930 Use verify_slice_dir subfun to get slice_dir even for a single file.
% 131001 dicm_hdr can deal with VR of SQ. This slows down it a little.
% 131002 Avoid fullfile for cellstr input (not supported in old ver matlab).
% 131006 Tweak dicm_hdr for multiframe dicom (some bug fixes);
%        First working version for multiframe (tested with Philips dicom).
% 131009 Put dicm_hdr, dicm_img, dicm_dict outside this file;
%        dicm_hdr can read implicit VR, and is faster with single fread;
%        Fix problem in gzipOS when folder name contains space.
% 131020 Make TR & ProtocolName non-mandatory; Set cal_min & cal_max.
% 131021 Check SamplesPerPixel, skip run if it is 1+.
% 131021 Implement conversion for AFNI HEAD/BRIK.
% 131024 Bug fix for dealing with current folder as src folder.
% 131029 Bug fix: Siemens, 2D, non-mosaic, rev-num slices were flipped.
% 131105 DTI parameters: field names more consistent; read DTI flds in
%        save_dti_para for GE/Philips (make others faster); convert Philips
%        bvec from deg into vector (need to be verified).
% 131114 Treak for multiframe dicm_hdr: MUCH faster by only 1,2,n frames;
%        Big fix for Philips multiframe DTI parameters;
%        Split multiframe Philips B0 map into mag and phase nii.
% 131117 Make the order of phase/mag image in Philips B0 map irrelevant.
% 131219 Write warning message to a file in data folder (Gui's suggestion).
% 140120 Bug fix in save_dti_para due to missing Manufacturer (Thank Paul).
% 140121 Allow missing instance at beginning.
% 140123 save_nii: bug fix for gzip.m detection, take care of ~ as home dir.
% 140206 bug fix: MoCo detetion bug introduced by removing empty cell earlier.
% 140223 add missing-file check for Philips data by slice locations.
% 140312 use slice timing to set slice_code for both GE and Siemens.
%         Interleaved order was wrong for GE data with even number of slices. 
% 140317 Use MosaicRefAcqTimes from last vol for multiband (thank Chris).
%        Don't re-orient fieldmap, so make FSL happy in case of non_axial. 
%        Ugly fix for wrong Siemens dicom item VR 'OB': Avoid using main header 
%         in csa_header(), convert DTI parameters to correct type. There may
%         be other wrong parameters we don't realize. 
% 140319 Store SliceTiming field in dcmHeaders.mat for FSL custom slice timing.
%        Re-orient even if flipping slices for 2D MRAcquisitionType.
% 140324 Not set cal_min, cal_max anymore.
% 140327 Return unconverted subject names in 2nd output.
% 140401 Always flip image so phase dir is correct.
% 140409 Store nii extension (not enabled).
% 140501 Fix for GE: use LocationsInAcquisition to replace ImagesInAcquisition;
%            isDTI=DiffusionDirection>0; Gradient already in image reference.
% 140505 Always re-orient DTI. bvec fix for GE DTI (thx Chris).
% 140506 Remove last DTI vol if it is computed ADC (as dcm2niix);
%        Use SeriesDescription to replace ProtocolName for file name;
%        Improved dim_info and phase direction.
% 140512 Decode GE ProtocolDataBlock for phase direction;
%        strtrim SeriesDescription for nii file name.
% 140513 change stored phase direction to image space for FSL unwarp;
%        Simplify code for dim_info.
% 140516 Switch back to ProtocolName for SIEMENS to take care of MOCO series;
%        Detect Philips Dim3IsVolume (for multi files) during dicom check; 
%        Work for GE interleaved slices even if InstanceNumber is in time order;
%        Do ImagePositionPatient check for all vendors;
%        Simplify code for save_dti_para.
% 140517 Store img with first dim flipped, to take care of DTI bvec problems. 
% 140522 Use SliceNormalVector for mosaic slice_dir, so no worry for revNumb;
%        Bug fix for interleaved descending slice_code.
% 140525 xform sliceCenter to SliceLocation in verify_slice_dir. 
% 140526 Take care of non-unique ixyz. 
% 140608 Bug fix for GE interleaved slices;
%        Take care all ixyz, put verify_slice_dir into xform_mat.
% 140610 Compute readout time for DTI, rather than dwell time.
% 140621 Support tgz file as data source.
% 140716 Take care of error of GUI subject option due to empty src.
% 140808 Simplify mosaic detection, and remove isMosaic.
% 140816 Simplify DTI detection.
% 140911 Minor fix for Siemens ProtocolName for error message.

if nargin>1 && ischar(src) && strcmp(src, 'dicm2nii_gui_cb')
    dicm2nii_gui(dataFolder); % mis-use first two input for GUI
    varargout = {'' ''};
    return;
end

%% Deal with output format first, and error out if invalid
if nargin<3 || isempty(varargin{1}), fmt = 1; % default .nii.gz
else fmt = varargin{1};
end

if (isnumeric(fmt) && (fmt==0 || fmt==1)) || ...
      (ischar(fmt) && ~isempty(regexpi(fmt, 'nii')))
    ext = '.nii';
elseif (isnumeric(fmt) && (fmt==2 || fmt==3)) || (ischar(fmt) && ...
        (~isempty(regexpi(fmt, 'hdr')) || ~isempty(regexpi(fmt, 'img'))))
    ext = '.img';
else
    error(' Invalid output file format (the 3rd input).');
end

if (isnumeric(fmt) && mod(fmt,2)) || ...
        (ischar(fmt) && ~isempty(regexpi(fmt, '.gz')))
    ext = [ext '.gz']; % gzip file
end

%% Deal with MoCo option
if nargin<4 || isempty(varargin{2})
    MoCo = 1; % by default, use original series if both present 
else
    MoCo = varargin{2};
    if ~any(MoCo==0:2)
        error(' Invalid MoCoOption. The 4th input must be 0, 1 or 2.');
    end
end

%% Deal with 5th input: we do one subject once
if nargin<5 || isempty(varargin{3})
    subjProvided = false; subj = '';
else 
    subjProvided = true; subj = varargin{3};
    if ~ischar(subj), error(' Invalid subject name.');end
end

%% Deal with data source
varargout = {};
unzip_cmd = '';
if nargin<1 || isempty(src) || (nargin<2 || isempty(dataFolder))
    create_gui; % show GUI if input is not enough
    return;
end

if isnumeric(src)
    error('Invalid dicom source.');    
elseif iscellstr(src) % multiple files
    dcmFolder = folderFromFile(src{1});
    n = length(src);
    fnames = src;
    for i = 1:n
        foo = dir(src{i});
        if isempty(foo), error('%s does not exist.', src{i}); end
        fnames{i} = fullfile(dcmFolder, foo.name); 
    end
elseif ~exist(src, 'file') % like input: run1*.dcm
    fnames = dir(src);
    if isempty(fnames), error('%s does not exist.', src); end
    fnames([fnames.isdir]) = [];
    dcmFolder = folderFromFile(src);
    fnames = strcat(dcmFolder, filesep, {fnames.name});    
elseif isdir(src) % folder
    dcmFolder = src;
elseif ischar(src) % 1 dicom or zip/tgz file
    dcmFolder = folderFromFile(src);
    unzip_cmd = compress_func(src);
    if isempty(unzip_cmd)
        fnames = dir(src);
        fnames = strcat(dcmFolder, filesep, {fnames.name});
    end
else 
    error('Unknown dicom source.');
end
dcmFolder = fullfile(getfield(what(dcmFolder), 'path'));

%% Deal with dataFolder
if nargin<2 || isempty(dataFolder)
    dataFolder = uigetdir(dcmFolder, 'Select a folder to save data files');
    if dataFolder==0, return; end
end
if ~isdir(dataFolder), mkdir(dataFolder); end
dataFolder = fullfile([getfield(what(dataFolder), 'path'), filesep]);
global dcm2nii_errFileName;
dcm2nii_errFileName = [dataFolder 'dicm2nii_warningMsg.txt'];

disp('Xiangrui Li''s dicm2nii (feedback to xiangrui.li@gmail.com)');

%% Unzip if compressed file is the source
tic;
if ~isempty(unzip_cmd)
    [~, fname, ext1] = fileparts(src);
    dcmFolder = sprintf('%stmpDcm%s/', dataFolder, fname);
    if ~isdir(dcmFolder), mkdir(dcmFolder); end
    disp(['Extracting files from ' fname ext1 ' ...']);

    if strcmp(unzip_cmd, 'unzip')
        cmd = sprintf('unzip -qq -o %s -d %s', src, dcmFolder);
        err = system(cmd); % first try system unzip
        if err, unzip(src, dcmFolder); end % Matlab's unzip is too slow
    elseif strcmp(unzip_cmd, 'untar')
        if isempty(which('untar')), error('No untar found in matlab path.'); end
        untar(src, dcmFolder);
    end
    drawnow;
end 

%% Get all file names including those in subfolders, if not specified
if ~exist('fnames', 'var')
    dirs = genpath(dcmFolder);
    dirs = textscan(dirs, '%s', 'Delimiter', pathsep);
    dirs = dirs{1}; % cell str
    fnames = {};
    for i = 1:length(dirs)
        curFolder = [dirs{i} filesep];
        foo = dir(curFolder); % all files and folders
        foo([foo.isdir]) = []; % remove folders
        foo = strcat(curFolder, {foo.name});
        fnames = [fnames foo]; %#ok<*AGROW>
    end
end
nFile = length(fnames);
if nFile<1, error(' No files found in the data source.'); end

%% Get Manufacturer
dict = dicm_dict('', 'Manufacturer');
vendor = '';
for i = unique([1 ceil(nFile*[0.2 0.5 0.8 1])]) % try up to 5 files
    s = dicm_hdr(fnames{i}, dict);
    if isempty(s), continue; end
    vendor = strtok(s.Manufacturer); % take 1st word only
    break;
end

%% Check each file, store header in cell array h
% first 6 fields are must for 1st round, next 3 are must for later check
flds = {'InstanceNumber' 'SeriesNumber' 'ImageType' 'Columns' 'Rows' ...
	'BitsAllocated' 'PixelSpacing' 'ImageOrientationPatient' ...
    'ImagePositionPatient' 'PixelRepresentation' 'SamplesPerPixel' ...
    'SeriesDescription' 'EchoTime' 'PatientName' 'PatientID' 'NumberOfFrames' ...
    'B_value' 'B_matrix' 'DiffusionGradientDirection' ...
    'RTIA_timer' 'TriggerTime' 'RBMoCoTrans' 'RBMoCoRot' };
dict = dicm_dict(vendor, flds); % get partial dict for the vendor
% Following for Philips only: B_value etc may be duplicated in later tags
ind = find(strcmp(dict.name, 'PixelRepresentation'), 1, 'last') + 1;
dict.name(ind:end) = []; dict.tag(ind:end) = []; dict.vr(ind:end) = []; 

junk = {'\MEAN' '\DUMMY IMAGE' '\TTEST' '\FMRI\DESIGN' ... % GLM
        '\DIFFUSION\ADC\' '\DIFFUSION\FA\' '\DIFFUSION\TRACEW\'}; % DTI

h = {}; % in case of no dicom files at all
subj_skip = {};
errInfo = '';
fprintf('Validating %g files (%s) ...\n', nFile, vendor);
if nFile > 2000
    fprintf('\b\b\b\b\b. May take about %.0f seconds ...\n', nFile*0.015);
end
for k = 1:nFile
    fname = fnames{k};
    [s, err] = dicm_hdr(fname, dict);
    if isempty(s) || any(~isfield(s, flds(1:6))) ...
         || isType(s,junk) || tryGetField(s, 'SamplesPerPixel', 1)>1
        errInfo = sprintf('%s\n%s\n', errInfo, err);
        continue; % skip the file
    end
    subj1 = tryGetField(s, 'PatientName');
    if isempty(subj1), subj1 = tryGetField(s, 'PatientID', 'unknown'); end
       
    % if not for single subject, do the first only
    if isempty(subj)
        subj = subj1; % store it for later check
    elseif ~strcmpi(subj, subj1)
        if nargout>1 && ~any(strcmp(subj_skip, subj1))
            subj_skip = [subj_skip {subj1}];
        end
        if ~subjProvided
            errorLog([fname ' is for a different subject ' subj1 '. Skipped.']);
        end
        continue;
    end

    % For fieldmap mag image, we use the one with short TE, which has
    % better quality. This also skips repeated copy of a file.
    i = s.SeriesNumber; j = s.InstanceNumber;
    try % ignore the error if the cell in h hasn't been filled.
       if s.EchoTime >= h{i}{j}.EchoTime, continue; end
    end
    
    % This fix supposes the multi-frame has only 1 file for each series
    if j<1, j = 1; end % InstanceNumber 0 violates dicom rule, but ...
    h{i}{j} = s; % store partial header
end
if nargout>0, varargout{1} = subj; end % return converted subject ID
if nargout>1, varargout{2} = unique(subj_skip); end % unconverted subject ID

%% Check headers: remove file-missing and dim-inconsistent series
nRun = length(h);
if nRun<1
    errorLog(sprintf('No valid files found for %s:\n%s.', subj, errInfo)); 
    return;
end
keep = true(1, nRun); % true for useful runs
isMoCo = false(1, nRun); % deal moco together later
for i = 1:nRun
    if isempty(h{i}), keep(i) = 0; continue; end % must be here due to MoCo
    ind = cellfun(@isempty, h{i});
    if any(ind) % there are empty cell(s)
        k = find(~ind, 1); 
        if strncmpi(tryGetField(h{i}{k},'Manufacturer'), 'Philips', 7)
            h{i} = h{i}(~ind);
        else % treat as missing file(s) if not at beginning
            h{i}(1:k-1) = []; % remove leading empty cells
            ind = cellfun(@isempty, h{i}); % still any empty?
            if any(ind)
                s = h{i}{find(ind, 1) - 1}; % the one before first empty cell
                errorLog(sprintf(['Series %g, %s, file after Instance %g is ' ...
                    'missing. Run skipped.'], s.SeriesNumber, ...
                    ProtocolName(s), s.InstanceNumber));
                keep(i) = 0; continue; % skip
            end
        end
    end
    
    if ~isfield(h{i}{1}, 'LastFile') % no re-read for PAR/AFNI file
        h{i}{1} = dicm_hdr(h{i}{1}.Filename); % full header for 1st file
    end
    h{i}{1} = multiFrameFields(h{i}{1}); % no-op if non multi-frame
    if ~isfield(h{i}{1}, 'Manufacturer'), h{i}{1}.Manufacturer = 'Unknown'; end
    s = h{i}{1};
    if any(~isfield(s, flds(7:9))), keep(i) = 0; continue; end
    isMoCo(i) = isType(s, '\MOCO\');
    nFrame = tryGetField(s, 'NumberOfFrames', 1);
    
    % check dimension, orientation, pixelSpacing consistency
    nFile = length(h{i});
    for j = 2:nFile
        s1 = h{i}{j};
        nFrame1 = tryGetField(s1, 'NumberOfFrames', 1);
        err1 = ~isequal([s.Columns s.Rows nFrame], [s1.Columns s1.Rows nFrame1]);
        err1 = err1 || sum(abs(s.PixelSpacing-s1.PixelSpacing)) > 1e-5;
        if err1 || (sum(abs(s1.ImageOrientationPatient - ...
               s.ImageOrientationPatient)) > 1e-5) % arbituary     
            errorLog(sprintf(['Inconsistent pixel size, image orientation ' ...
             'and/or dimension for subject %s, Series %g, %s. Run skipped.\n'], ...
             subj, s.SeriesNumber, ProtocolName(s)));
            keep(i) = 0; break; % skip
        end
    end

    % this won't catch all missing files, but catch most cases.
    if ~keep(i) || nFile<2, continue; end
    iSL = xform_mat(s); iSL = iSL(3);
    a = zeros(1, nFile);
    for j = 1:nFile, a(j) = h{i}{j}.ImagePositionPatient(iSL); end
    if mod(nFile, numel(unique(a)))>0 % may be too strict to be equal?
        errorLog(sprintf(['Series %g, %s, seems file(s) missing. ' ...
            'Run skipped.'], s.SeriesNumber, ProtocolName(s)));
        keep(i) = 0; continue; % skip
    end
    
    if strncmpi(s.Manufacturer, 'GE', 2)
        % re-order slices within volume by slice locations: not tested
        nSL = double(tryGetField(s, 'LocationsInAcquisition', nFile));
        [foo, ind] = sort(a(1:nSL)); % slice locations for first volume
        if isequal(ind, 1:nSL) || isequal(ind, nSL:-1:1), continue; end
        if foo(1) ~= a(1), ind = ind(nSL:-1:1); end
        nVol = nFile / nSL;
        inc = repmat((0:nVol-1)*nSL, nSL, 1);
        ind = repmat(ind, 1, nVol) + inc(:)';
        h{i} = h{i}(ind); % sorted by slice locations
    elseif strncmpi(s.Manufacturer, 'Philips', 7)
        % re-order as Slice then Volume if Dim3IsVolume
        if a(1) ~= a(2), continue; end
        nSL = tryGetField(s, 'SlicesPerVolume', nFile);
        nVol = nFile / nSL;
        ind = reshape(1:nFile, [nVol nSL])';
        h{i} = h{i}(ind(:));
        h{i}{1}.Dim3IsVolume = true; % for info only
    end
end

ind = find(isMoCo); % decide MoCo after checking all series
for i = 1:length(ind)
    if MoCo==1 && keep(ind(i)-1) % in case original skipped, keep MOCO
        keep(ind(i)) = 0; continue; % skip MOCO
    elseif MoCo==2 && keep(ind(i)) % in case MOCO skipped, keep original
        keep(ind(i)-1) = 0; % skip previous series (original)
    end
end
h = h(keep); % remove all unwanted series once

%% Generate unique file names
% Unique names are in format of SeriesDescription_s007. Special cases are: 
%  for phase image, such as field_map phase, append '_phase' to the name;
%  for MoCo series, append '_MoCo' to the name if both series are created.
nRun = length(h); % update it, since we may have removed some
if nRun<1
    errorLog(['No valid series found for ' subj]);
    return;
end
rNames = cell(1, nRun);
for i = 1:nRun
    s = h{i}{1};
    a = strtrim(ProtocolName(s));
    a(~isstrprop(a, 'alphanum')) = '_'; % make str valid for field name
    while true % remove repeated underscore
        ind = strfind(a, '__');
        if isempty(ind), break; end
        a(ind+1) = '';
    end
    if isType(s, '\P\'), a = [a '_phase']; end % phase image
    if MoCo==0 && isType(s, '\MOCO\'), a = [a '_MoCo']; end
    sN = s.SeriesNumber;
    if sN>100 && strncmp(s.Manufacturer, 'Philips', 7)
        sN = tryGetField(s, 'AcquisitionNumber', floor(sN/100));
    end
    rNames{i} = sprintf('%s_s%03g', a, sN);
end
rNames = genvarname(rNames); % add 'x' if started with a digit, and more

% After following sort, we need to compare only neighboring names. Remove
% _s007 if there is no conflict. Have to ignore letter case for Windows & MAC
fnames = rNames; % copy it, keep letter cases
[rNames, iRuns] = sort(lower(rNames)); 
for i = 1:nRun
    a = rNames{i}(1:end-5); % remove _s003
    % no conflict with both previous and next name
    if nRun==1 || ... % only one run
         (i==1    && ~strcmpi(a, rNames{2}(1:end-5))) || ... % first
         (i==nRun && ~strcmpi(a, rNames{i-1}(1:end-5))) || ... % last
         (i>1 && i<nRun && ~strcmpi(a, rNames{i-1}(1:end-5)) ...
         && ~strcmpi(a, rNames{i+1}(1:end-5))); % middle ones
        fnames{iRuns(i)}(end+(-4:0)) = [];
    end
end
fmtStr = sprintf(' %%-%gs %%4g\n', max(cellfun(@length, fnames))+6);

%% Now ready to convert nii run by run
fprintf('Converting %g series into %s: subject %s\n', nRun, ext, subj);
for i = 1:nRun
    nFile = length(h{i});
    fprintf(fmtStr, fnames{i}, nFile); % show info and progress
    h{i}{1}.isDTI = isDTI(h{i}{1});
    s = h{i}{1};
    
    img = dicm_img(s); % initialize with proper data type and img size

    if nFile > 1
        h{i}{1}.LastFile = h{i}{nFile}; % store partial last header into 1st
        n = ndims(img);
        if n == 2
            img(:, :, 2:nFile) = 0; % pre-allocate
            for j = 2:nFile, img(:,:,j) = dicm_img(h{i}{j}); end
        elseif n == 3 % if one file is for one vol
            img(:, :, :, 2:nFile) = 0; % pre-allocate
            for j = 2:nFile, img(:,:,:,j) = dicm_img(h{i}{j}); end
        else % err out, likely won't work for other series
            error('dicm2nii can''t deal with %g-dim dicom image', n);
        end
    end
    
    if ~isempty(csa_header(s, 'NumberOfImagesInMosaic')) % SIEMENS mosaic
        img = mos2vol(img, s); % mosaic to volume
    elseif ndims(img)==4 && tryGetField(s, 'Dim3IsVolume', false)
        img = permute(img, [1 2 4 3]);
    elseif ndims(img) == 3 % get nSlice for different situation
        if strncmpi(vendor, 'SIEMENS', 7)
            if strcmp(tryGetField(s, 'MRAcquisitionType'), '3D')
                nSL = asc_header(s, 'sKSpace.lImagesPerSlab');
            else
                nSL = asc_header(s, 'sSliceArray.lSize');
            end
        else % GE and Philips
            nSL = tryGetField(s, 'SlicesPerVolume');
            if isempty(nSL), nSL= tryGetField(s, 'LocationsInAcquisition'); end
        end
        
        if nSL>1
            dim = size(img);
            dim(3:4) = [nSL dim(3)/double(nSL)];
            if nFile==1 && tryGetField(s, 'Dim3IsVolume', false)
                % for PAR and single multiframe dicom
                img = reshape(img, dim([1 2 4 3]));
                img = permute(img, [1 2 4 3]);
            else
                img = reshape(img, dim);
            end
        end
    end
    dim = size(img);
    if numel(dim)<3, dim(3) = 1; end
        
    % Store GE slice timing. No slice order info for Philips at all!
    flds = {'RTIA_timer' 'TriggerTime'};
    ind = find(isfield(s, flds), 1);
    if ~isempty(ind)
        t = zeros(dim(3), 1);
        for j = 1:dim(3), t(j) = tryGetField(h{i}{j}, flds{ind}, nan); end
        h{i}{1}.SliceTiming = t/10; % in ms
    end
    
    % Store motion parameters for MoCo series (assume it is mosaic)
    if all(isfield(s, {'RBMoCoTrans' 'RBMoCoRot'}))
        trans = zeros(nFile, 3);
        rotat = zeros(nFile, 3);
        for j = 1:nFile
            trans(j,:) = tryGetField(h{i}{j}, 'RBMoCoTrans', [0 0 0]);
            rotat(j,:) = tryGetField(h{i}{j}, 'RBMoCoRot', [0 0 0]);
        end
        h{i}{1}.RBMoCoTrans = trans;
        h{i}{1}.RBMoCoRot = rotat;
    end
    
    if isa(img, 'uint16') && max(img(:))<32768
        img = int16(img); % use int16 if lossless. seems always true
    end
    
    nii = make_nii(img); % need NIfTI toolbox
    fname = [dataFolder fnames{i}];
    % Save FSL bval and bvec files for DTI data
    if s.isDTI, [h{i}, nii] = save_dti_para(h{i}, nii, fname); end
    [nii, h{i}{1}] = set_nii_header(nii, h{i}{1}); % set most nii header
    nii = save_philips_phase(nii, s, dataFolder, fnames{i}, ext, fmtStr);
    save_nii(nii, [fname ext]); % need NIfTI toolbox

    h{i} = h{i}{1}; % keep 1st dicm header only
end

h = cell2struct(h, fnames, 2); % convert into struct
fname = [dataFolder 'dcmHeaders.mat'];
if exist(fname, 'file') % if file exists, we update fields only
    S = load(fname);
    for i = 1:length(fnames), S.h.(fnames{i}) = h.(fnames{i}); end
    h = S.h; %#ok
end
save(fname, 'h', '-v7'); % -v7 better compatibility
fprintf('Elapsed time by dicm2nii is %.1f seconds\n\n', toc);
if ~isempty(unzip_cmd), rmdir(dcmFolder, 's'); end % delete tmp dicom folder
return;

%% Subfunction: return folder name for a file name
function folder = folderFromFile(fname)
folder = fileparts(fname);
if isempty(folder), folder = pwd; end

%% Subfunction: return SeriesDescription
function name = ProtocolName(s)
name = tryGetField(s, 'SeriesDescription');
if strncmpi(s.Manufacturer, 'SIEMENS', 7) || isempty(name)
    name = tryGetField(s, 'ProtocolName', name); 
end
if isempty(name), [~, name] = fileparts(s.Filename); end

%% Subfunction: return true if any of keywords is in s.ImageType
function tf = isType(s, keywords)
keywords = cellstr(keywords);
for i = 1:length(keywords)
    key = strrep(keywords{i}, '\', '\\'); % for regexp
    tf = ~isempty(regexp(s.ImageType, key, 'once'));
    if tf, return; end
end

%% Subfunction: return true if it is DTI
function tf = isDTI(s)
tf = isType(s, '\DIFFUSION'); % Siemens, Philips
if tf, return; end
if strncmp(s.Manufacturer, 'GE', 2) % not labeled as /DIFFISION
    tf = tryGetField(s, 'DiffusionDirection', 0)>0;
elseif strncmpi(s.Manufacturer, 'Philips', 7)
    tf = strcmp(tryGetField(s, 'MRSeriesDiffusion', 'N'), 'Y');
else % Some Siemens DTI are labeled as \DIFFUSION
    tf = ~isempty(csa_header(s, 'B_value'));
end
        
%% Subfunction: get field if exist, return default value otherwise
function val = tryGetField(s, field, dftVal)
if isfield(s, field), val = s.(field); 
elseif nargin>2, val = dftVal;
else val = [];
end

%% Subfunction: Set most nii header and re-orient img. 
function [nii, s] = set_nii_header(nii, s)
TR = tryGetField(s, 'RepetitionTime');
if isempty(TR), TR = tryGetField(s, 'TemporalResolution', 2000); end
dim = nii.hdr.dime.dim(2:4); % image dim, set by make_nii according to img

% Transformation matrix: most important feature for nii
[ixyz, R, pixdim] = xform_mat(s, dim);
R(1:2,:) = -R(1:2,:); % dicom LPS to nifti RAS, xform matrix before reorient

% Flip image to make major axis positive
ind4 = ixyz + [0 4 8]; % index in 4xN matrix
flip = R(ind4)<0; % flip an axis if true
rotM = diag([sign(0.5-flip) 1]); % 1 or -1 on diagnal
rotM(1:3, 4) = (dim-1) .* flip;
R = R / rotM; % xform matrix after flip
for k = 1:3, if flip(k), nii.img = flipdim(nii.img, k); end; end

% dim_info byte: freq_dim, phase_dim, slice_dim low to high, each 2 bits
fps_bits = [1 4 16]; % for axes 1 2 3: freq_dim, phase_dim, slice_dim
[phPos, iPhase] = phaseDirection(s); % relative to image in FSL feat!
if iPhase==ixyz(1), fps_bits = [4 1 16]; end

% Reorient if MRAcquisitionType==3D || isDTI
% If FSL etc can read dim_info for STC, we can always reorient.
[~, perm] = sort(ixyz); % may permute 3 dimensions in this order
if (strcmp(tryGetField(s, 'MRAcquisitionType'), '3D') || s.isDTI) ...
        && (~isequal(perm, 1:3)) % skip if already standard view
    R(:, 1:3) = R(:, perm); % xform matrix after perm
    fps_bits = fps_bits(perm);
    ixyz = ixyz(perm); % 1:3 after re-orient
    nii.hdr.dime.dim(2:4) = dim(perm);
    pixdim = pixdim(perm);
    nii.img = permute(nii.img, [perm 4]); % permute img after flip
end

% flip first axis after possible perm (added these lines on 20140517)
rotM = diag([-1 1 1 1]);
rotM(1,4) = nii.hdr.dime.dim(2)-1;
R = R / rotM;
nii.img = flipdim(nii.img, 1);
if ixyz(1) == iPhase, phPos = ~phPos; end

frmCode = tryGetField(s, 'TemplateSpace', 1); % 1: SCANNER_ANAT
nii.hdr.hist.sform_code = frmCode;
nii.hdr.hist.srow_x = R(1,:);
nii.hdr.hist.srow_y = R(2,:);
nii.hdr.hist.srow_z = R(3,:);

nii.hdr.hist.qform_code = frmCode;
nii.hdr.hist.qoffset_x = R(1,4);
nii.hdr.hist.qoffset_y = R(2,4);
nii.hdr.hist.qoffset_z = R(3,4);

R = R(1:3, 1:3); % for quaternion
R = R ./ repmat(sqrt(sum(R.^2)),3,1); % avoid normc from nnet toolbox
proper = sign(det(R)); % always -1 if reorient & flip_dim1
if proper<0, R(:,3) = -R(:,3); end
nii.hdr.dime.pixdim(1) = proper; % -1 or 1 

q = dcm2quat(R); % 3x3 dir cos matrix to quaternion
if q(1)<0, q = -q; end % as MRICron
nii.hdr.hist.quatern_b = q(2);
nii.hdr.hist.quatern_c = q(3);
nii.hdr.hist.quatern_d = q(4);

str = tryGetField(s, 'ImageComments');
if isType(s, '\MOCO\'), str = ''; end % useless for MoCo
foo = tryGetField(s, 'StudyComments');
if ~isempty(foo), str = [str ';' foo]; end
str = [str ';' strtok(s.Manufacturer)];
foo = tryGetField(s, 'ProtocolName');
if ~isempty(foo), str = [str ';' foo]; end
nii.hdr.hist.aux_file = str; % char[24], info only
seq = asc_header(s, 'tSequenceFileName'); % like '%SiemensSeq%\ep2d_bold'
[~, seq] = strtok(seq, '\'); seq = strtok(seq, '\'); % like 'ep2d_bold'
if isempty(seq), seq = tryGetField(s, 'ScanningSequence'); end
id = tryGetField(s, 'PatientID');
nii.hdr.hk.db_name = [seq ';' id]; % char[18], optional

% save some useful info in descrip: info only
foo = tryGetField(s, 'AcquisitionDateTime');
if isempty(foo) 
    foo = tryGetField(s, 'AcquisitionDate');
    foo = [foo tryGetField(s, 'AcquisitionTime')];
end
descrip = sprintf('time=%s;', foo(1:min(18,end))); 
TE0 = asc_header(s, 'alTE[0]')/1000; % s.EchoTime stores only 1 TE
dTE = asc_header(s, 'alTE[1]')/1000 - TE0; % TE difference for fieldmap
if isempty(TE0), TE0 = tryGetField(s, 'EchoTime'); end % GE, philips
if isempty(dTE) && tryGetField(s, 'NumberOfEchoes', 1)>1
    dTE = tryGetField(s, 'SecondEchoTime') - TE0; % need to update
end
if ~isempty(dTE)
    descrip = sprintf('dTE=%.4g;%s', abs(dTE), descrip);
    s.deltaTE = dTE;
elseif ~isempty(TE0)
    descrip = sprintf('TE=%.4g;%s', TE0, descrip);
end

% Get dwell time, slice timing info
if ~strcmp(tryGetField(s, 'MRAcquisitionType'), '3D')
    hz = csa_header(s, 'BandwidthPerPixelPhaseEncode');
    dwell = 1000 ./ hz / dim(iPhase==ixyz); % in ms
    if isempty(dwell) % true for syngo MR 2004A
        % ppf = [1 2 4 8 16] represent [4 5 6 7 8] 8ths PartialFourier
        % ppf = asc_header(s, 'sKSpace.ucPhasePartialFourier');
        lns = asc_header(s, 'sKSpace.lPhaseEncodingLines');
        dur = csa_header(s, 'SliceMeasurementDuration');
        dwell = dur ./ lns; % ./ (log2(ppf)+4) * 8;
    end
    if isempty(dwell) % next is not accurate, so as last resort
        dur = csa_header(s, 'RealDwellTime') * 1e-6; % ns to ms
        dwell = dur * asc_header(s, 'sKSpace.lBaseResolution');
    end
    if isempty(dwell)
        dwell = double(tryGetField(s, 'EffectiveEchoSpacing')) / 1000; % GE
    end
    % http://www.spinozacentre.nl/wiki/index.php/NeuroWiki:Current_developments
    if isempty(dwell) % philips
        wfs = tryGetField(s, 'WaterFatShift');
        epiFactor = tryGetField(s, 'EPIFactor');
        dwell = wfs ./ (434.215 * (double(epiFactor)+1)) * 1000;
    end
    if ~isempty(dwell)
        if s.isDTI
            readout = dwell * dim(iPhase==ixyz) / 1000; % in sec
            descrip = sprintf('readout=%.3g;%s', readout, descrip);
            s.ReadoutSeconds = readout;
        else
            descrip = sprintf('dwell=%.3g;%s', dwell, descrip);
            s.EffectiveEPIEchoSpacing = dwell;
        end
    end
    
    t = csa_header(s, 'MosaicRefAcqTimes'); % in ms
    % MosaicRefAcqTimes for first vol may be wrong for Siemens MB
    if ~isempty(t) && isfield(s, 'LastFile') % pity: no MB flag in dicom
        dict = dicm_dict(s.Manufacturer, 'MosaicRefAcqTimes');
        s2 = dicm_hdr(s.LastFile.Filename, dict);
        t = s2.MosaicRefAcqTimes; % to be safe, use the last file
    end
    if isempty(t), t = tryGetField(s, 'SliceTiming'); end % for GE
    
    if numel(t)>1 % 1+ slices
        t1 = sort(t);
        dur = mean(diff(t1));
        dif = mean(diff(t));
        if dur==0 || (t1(end)>TR), sc = 0; % no useful info, or bad timing MB
        elseif t1(1) == t1(2), sc = 7; % timing available MB, made-up code 7
        elseif abs(dif-dur)<1e-3, sc = 1; % ascending
        elseif abs(dif+dur)<1e-3, sc = 2; % descending
        elseif t(1)<t(3) % ascending interleaved
            if t(1)<t(2), sc = 3; % odd slices first
            else sc = 5; % Siemens even number of slices
            end
        elseif t(1)>t(3) % descending interleaved
            if t(1)>t(2), sc = 4;
            else sc = 6; % Siemens even number of slices
            end
        else sc = 0; % unlikely to reach
        end
 
        if xor(flip(3), fps_bits(1) == 16) % assume we flip dim1 earlier
            if sc>0, sc = sc+mod(sc,2)*2-1; end % 1<->2, 3<->4, 5<->6
            t = t(end:-1:1);
        end
        t = t - min(t); % it may be relative to 1st slice
        if sc>0, s.SliceTiming = 0.5 - t/TR; end % as for FSL custom timing
        nii.hdr.dime.slice_code = sc;
    end
    nii.hdr.dime.slice_end = dim(3)-1; % 0-based, slice_start default to 0

    dur = min(diff(sort(t))); % 2.5ms error Siemens, dur = 0 for MB
    if isempty(dur) % in case MosaicRefAcqTimes is not available
        delay = asc_header(s, 'lDelayTimeInTR')/1000; % in ms now
        if isempty(delay), delay = 0; end
        dur = (TR-delay)/dim(3); 
    end
    nii.hdr.dime.slice_duration = dur/1000; 
end

% data slope and intercept: apply to img if no rounding error 
nii.hdr.dime.scl_slope = 1; % default scl_inter is 0
if any(isfield(s, {'RescaleSlope' 'RescaleIntercept'}))
    slope = tryGetField(s, 'RescaleSlope', 1); 
    inter = tryGetField(s, 'RescaleIntercept', 0); 
    val = sort([nii.hdr.dime.glmax nii.hdr.dime.glmin] * slope + inter);
    dClass = class(nii.img);
    if isa(nii.img, 'float') || (mod(slope,1)==0 && mod(inter,1)==0 ... 
            && val(1)>=intmin(dClass) && val(2)<=intmax(dClass))
        nii.img = nii.img * slope + inter; % apply to img if no rounding
    else
        nii.hdr.dime.scl_slope = slope;
        nii.hdr.dime.scl_inter = inter;
    end
end

if isempty(phPos), pm = '?'; elseif phPos, pm = ''; else pm = '-'; end
axes = 'xyz';
phDir = [pm axes(ixyz == iPhase)];
s.UnwarpDirection = phDir;
descrip = sprintf('phase=%s;%s', phDir, descrip);

nii.hdr.hist.descrip = descrip; % char[80], drop from end if exceed
nii.hdr.hk.dim_info = (1:3) * fps_bits'; % useful for EPI only
nii.hdr.dime.xyzt_units = 10; % mm (2) + seconds (8)
nii.hdr.dime.pixdim(2:5) = [pixdim TR/1000]; % voxSize and TR

% Possible patient position: HFS/HFP/FFS/FFP / HFDR/HFDL/FFDR/FFDL
% Seems dicom takes care of this, and maybe nothing needs to do here.
% patientPos = tryGetField(s, 'PatientPosition', 'HFS');

% Save 1st dicom hdr into nii extension
% try
%     s.nii_creator = ['dicm2nii.m ' datestr(now, 'yyyymmdd')];
%     fname = [tempname '.mat'];
%     save(fname, '-struct', 's', '-v7'); % field as variable
%     fid = fopen(fname);
%     b = fread(fid, '*uint8')'; % data bytes
%     fclose(fid);
%     delete(fname);
% 
%     % first 4 bytes (int32) encode real data length
%     nii.ext.section.edata = [typecast(int32(numel(b)), 'uint8') b];
%     nii.ext.section.ecode = 40; % Matlab
% end

% nii.ext.section.ecode = 2; % dicom
% fid = fopen(s.Filename);
% nii.ext.section.edata = fread(fid, s.PixelData.Start, '*uint8');
% fclose(fid);

return;
% hdr.hist.magic, glmax, glmin will be taken care of by save_nii.
% magic: 'ni1', hdr/img pair; 'n+1', single nii file, empty for ANALYZE. 
% Not used: char data_type[10], char regular

%% Subfunction, reshape mosaic into volume, remove padded zeros
function img = mos2vol(img, s)
nSL = csa_header(s, 'NumberOfImagesInMosaic'); % not work for some data
if isempty(nSL), nSL = asc_header(s, 'sSliceArray.lSize'); end % useless
nMos = ceil(sqrt(nSL)); % always nMos x nMos tiles
[nc, nr, nv] = size(img); % number of col, row and vol
sz = [nc nr] / nMos; % slice size

% Get index in vol for one mosaic: not elegant, but brief
[rm, cm] = ind2sub([nc nr], 1:nc*nr); % row, col sub in mosaic
rv = mod(rm-1, sz(1)) + 1; % row index in vol
cv = mod(cm-1, sz(2)) + 1; % col index in vol
sv = floor((rm-1)/sz(1))+1 + floor((cm-1)/sz(2))*nMos; % slice index
iv = sub2ind([sz nMos^2], rv, cv, sv); % singlar index in vol

img = reshape(img, [nc*nr nv]); % one col per mosaic
img(iv, :) = img; % change to vol order
img = reshape(img, [sz nMos^2 nv]); % vol now
img(:, :, nSL+1:end, :) = []; % remove padded slices        

%% subfunction: extract bval & bvec, save in 1st header and files
function [h, nii] = save_dti_para(h, nii, fname)
dim = nii.hdr.dime.dim(4:5);
nDir = dim(2);
bval = nan(nDir, 1);
bvec = nan(nDir, 3);
bmtx = nan(nDir, 6);
nFile = length(h);
s = h{1};

if tryGetField(s, 'IsPhilipsPAR', false) || tryGetField(s, 'IsAFNIHEAD', false)
    bval = s.B_value;
    bvec = tryGetField(s, 'DiffusionGradientDirection', nan(nDir, 3));
elseif isfield(s, 'PerFrameFunctionalGroupsSequence')
    fld = 'PerFrameFunctionalGroupsSequence';
    if tryGetField(s, 'Dim3IsVolume', false)
        inc = 1; iFrames = 1:nDir;
    else
        inc = dim(1); iFrames = 1:dim(1):dim(1)*dim(2);
    end
    dict = dicm_dict(s.Manufacturer, {fld 'B_value' 'MRDiffusionSequence' ...
        'DiffusionGradientDirectionSequence' 'DiffusionGradientDirection'});
    s2 = dicm_hdr(s.Filename, dict, iFrames); % re-read needed frames
    sq = s2.(fld);
    for j = 1:nDir
        item = sprintf('Item_%g', (j-1)*inc+1);
        try
            a = sq.(item).MRDiffusionSequence.Item_1;
            bval(j) = a.B_value;
            a = a.DiffusionGradientDirectionSequence.Item_1;
            bvec(j,:) = a.DiffusionGradientDirection;
        end
    end
else
    if nFile == nDir, inc = 1; % mosaic
    else inc = dim(1); % file order already in slices then volumes
    end
    dict = dicm_dict(s.Manufacturer, {'B_value' 'SlopInt_6_9' 'B_matrix' ...
       'DiffusionDirectionX' 'DiffusionDirectionY' 'DiffusionDirectionZ'});
    for j = 1:nDir % no these values for 1st file of each excitation
        k = inc*(j-1) + 1;
        s2 = h{k};
        vec = tryGetField(s2, 'DiffusionGradientDirection');
        if isempty(vec) % GE, Philips
            s2 = dicm_hdr(s2.Filename, dict);
            vec(1) = tryGetField(s2, 'DiffusionDirectionX', 0);
            vec(2) = tryGetField(s2, 'DiffusionDirectionY', 0);
            vec(3) = tryGetField(s2, 'DiffusionDirectionZ', 0);
        end
        bvec(j,:) = vec;
        
        val = tryGetField(s2, 'B_value');
        if isempty(val) && isfield(s2, 'SlopInt_6_9') % GE
            val = s2.SlopInt_6_9(1);
        end
        if isempty(val), val = 0; end % B0
        bval(j) = val;
        bmtx(j,:) = tryGetField(s2, 'B_matrix', nan(1, 6));        
    end
end

if all(isnan(bval)) && all(isnan(bvec(:)))
    errorLog(['Failed to get DTI parameters: ' ProtocolName(s)]);
    return; 
end
bval(isnan(bval)) = 0;
bvec(isnan(bvec)) = 0;

h{1}.B_value = bval; % store all into header of 1st file
h{1}.DiffusionGradientDirection = bvec; % original
if ~all(isnan(bmtx(:))), h{1}.B_matrix = bmtx; end

if strncmpi(s.Manufacturer, 'Philips', 7)
    if max(sum(bvec.^2, 2)) > 2 % guess in degree
        for j = 1:nDir, bvec(j,:) = ang2vec(bvec(j,:)); end % deg to dir cos mat
        errorLog(['Please validate bvec (direction in deg): ' ProtocolName(s)]);
    end
    if bval(nDir)~=0 && all(abs(bvec(nDir,:))<1e-4)
        % Remove last vol if it is computed ADC
        bval(nDir) = [];
        bvec(nDir,:) = [];
        nii.img(:,:,:,nDir) = [];
        nDir = nDir - 1;
        nii.hdr.dime.dim(5) = nDir;
    end
end

% http://wiki.na-mic.org/Wiki/index.php/NAMIC_Wiki:DTI:DICOM_for_DWI_and_DTI
if strncmp(s.Manufacturer, 'GE', 2) % GE bvec already in image reference
    if strcmp(tryGetField(s, 'InPlanePhaseEncodingDirection'), 'ROW')
        bvec = bvec(:, [2 1 3]);
        bvec(:,3) = -bvec(:,3);
    else
        bvec(:,1) = -bvec(:,1);
    end
else
    R = reshape(s.ImageOrientationPatient, 3, 2);
    R(:,3) = null(R');
    bvec = bvec * R; % dicom plane to image plane
    bvec(:, 2) = -bvec(:, 2);
end

ixyz = xform_mat(s);
if ixyz(3) < 3 % sign change due to re-orient
    if ixyz(3) == 2 % COR, based on SIEMENS dataset
        bvec(:, 3) = -bvec(:, 3);
    elseif ixyz(3) == 1 % SAG, based on SIEMENS dataset
        bvec(:, 2) = -bvec(:, 2);
    end
    [~, perm] = sort(ixyz);
    bvec = bvec(:, perm); % reflect possible re-orient
end
h{1}.bvec = bvec; % store computed bvec in nii extension

fid = fopen([fname '.bval'], 'w');
fprintf(fid, '%g ', bval); % one row
fclose(fid);

str = repmat('%.15g ', 1, nDir);
fid = fopen([fname '.bvec'], 'w');
fprintf(fid, [str '\n'], bvec); % 3 rows by # direction cols
fclose(fid);

%% Subfunction: convert rotation angles to vector
function vec = ang2vec(ang)
% do the same as in philips_par: not sure it is right
ca = cosd(ang); sa = sind(ang);
rx = [1 0 0; 0 ca(1) -sa(1); 0 sa(1) ca(1)]; % standard 3D rotation
ry = [ca(2) 0 sa(2); 0 1 0; -sa(2) 0 ca(2)];
rz = [ca(3) -sa(3) 0; sa(3) ca(3) 0; 0 0 1];
R = rx * ry * rz;
% [~, iSL] = max(abs(R(:,3)));
% if iSL == 1 % Sag
%     R(:,[1 3]) = -R(:,[1 3]);
%     R = R(:, [2 3 1]);
% elseif iSL == 2 % Cor
%     R(:,3) = -R(:,3);
%     R = R(:, [1 3 2]);
% end
% http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
vec = [R(3,2)-R(2,3); R(1,3)-R(3,1); R(2,1)-R(1,2)];
vec = vec / sqrt(sum(vec.^2));

%% Subfunction, return a parameter from CSA header
function val = csa_header(s, key)
val = [];
csa = 'CSAImageHeaderInfo';
if ~isfield(s, csa), return; end % non-SIEMENS
if isstruct(s.(csa))
    if isfield(s.(csa), key)
        val = s.(csa).(key);
    end
    return;
end

% following can be reached only if CSA decoding failed in dicm_hdr.m
csa = s.(csa)';
ind = strfind(csa, [0 key 0]); % only whole word, offset by -1
val = {};
for j = 1:length(ind) % if keyword repeated, try each till we get val
    i0 = ind(j) + 68; % skip name vm (64+4)
    vr = char(csa(i0+(1:2))); i0=i0+8; % skip vr syngodt (4+4)
    n = typecast(csa(i0+(1:4)), 'int32'); % number of items
    if n<=0 || n>512, continue; end % give up if weird, 512 arbituary
    i0 = i0 + 8; % skip nitems xx (4+4)
    for i = 1:n % often times, n=6, but only 1st has len>0
        len = typecast(csa(i0+(1:4)), 'int32'); % # of bytes for item
        if len<=0 || len>512, break; end
        i0 = i0 + 16; % skip len len & int32 * 2 junk
        val{i,1} = char(csa(i0+(1:len-1))); % drop null
        i0 = i0 + ceil(double(len)/4)*4; % multiple 4-byte
    end
    if ~isempty(val), break; end
end

if isempty(strfind('AE AS CS DA DT LO LT PN SH ST TM UI UN UT', vr))
    val = str2double(val); % numeric as double
else
    val = val{1};
end

%% Subfunction, Convert 3x3 direction cosine matrix to quaternion
% Simplied from Quaternions by Przemyslaw Baranski 
function q = dcm2quat(R)
q = zeros(1, 4);
q(1) = sqrt(1 + R(1,1) + R(2,2) + R(3,3)) / 2;
q(2) = sqrt(1 + R(1,1) - R(2,2) - R(3,3)) / 2;
q(3) = sqrt(1 - R(1,1) + R(2,2) - R(3,3)) / 2;
q(4) = sqrt(1 - R(1,1) - R(2,2) + R(3,3)) / 2;
if ~isreal(q(1)), q(1) = 0; end % if trace(R)+1<0, zero it
[m, ind] = max(q);

switch ind
    case 1
        q(2) = (R(3,2) - R(2,3)) /m/4;
        q(3) = (R(1,3) - R(3,1)) /m/4;
        q(4) = (R(2,1) - R(1,2)) /m/4;
    case 2
        q(1) = (R(3,2) - R(2,3)) /m/4;
        q(3) = (R(1,2) + R(2,1)) /m/4;
        q(4) = (R(3,1) + R(1,3)) /m/4;
    case 3
        q(1) = (R(1,3) - R(3,1)) /m/4;
        q(2) = (R(1,2) + R(2,1)) /m/4;
        q(4) = (R(2,3) + R(3,2)) /m/4;
    case 4
        q(1) = (R(2,1) - R(1,2)) /m/4;
        q(2) = (R(3,1) + R(1,3)) /m/4;
        q(3) = (R(2,3) + R(3,2)) /m/4;
end

%% Subfunction: get dicom xform matrix and related info
function [ixyz, R, pixdim] = xform_mat(s, dim)
R = reshape(s.ImageOrientationPatient, 3, 2);
R(:, 3) = null(R');
foo = abs(R);
[~, ixyz] = max(foo); % orientation info: perm of 1:3
if ixyz(2) == ixyz(1), foo(ixyz(2),2) = 0; [~, ixyz(2)] = max(foo(:,2)); end
if ixyz(3) == ixyz(1), foo(ixyz(3),3) = 0; [~, ixyz(3)] = max(foo(:,3)); end
if ixyz(3) == ixyz(2), foo(ixyz(3),3) = 0; [~, ixyz(3)] = max(foo(:,3)); end
if nargout<2, return; end

iSL = ixyz(3); % 1/2/3 for Sag/Cor/Tra slice
thk = tryGetField(s, 'SpacingBetweenSlices');
if isempty(thk), thk = tryGetField(s, 'SliceThickness'); end
if isempty(thk) % this may never happen
    try
        thk = s.LastFile.ImagePositionPatient - s.ImagePositionPatient;
        thk = abs(thk(iSL) / R(iSL,3)) / (dim(3)-1);
    catch
        errorLog(['No slice thickness information found: ' ProtocolName(s)]);
        thk = s.PixelSpacing(1); % guess it is cubic
    end
end
pixdim = [s.PixelSpacing(:)' thk];
R = R * diag(pixdim); % apply vox size
% Next is almost dicom xform matrix, except mosaic trans and unsure slice_dir
R = [R s.ImagePositionPatient; 0 0 0 1];

% rest are former: R = verify_slice_dir(R, s, dim, iSL)
if dim(3) < 2, return; end % don't care direction for single slice
pos = []; % SliceLocation for last or center slice we try to retrieve

if s.Columns > dim(1) % Siemens mosaic is special case
    R(:,4) = R * [ceil(sqrt(dim(3))-1)*dim(1:2)/2 0 1]'; % transform back
    vec = csa_header(s, 'SliceNormalVector'); % mosaic has this
    if ~isempty(vec) % 100% exist for all tested data
        if sign(vec(iSL)) ~= sign(R(iSL,3)), R(:,3) = -R(:,3); end
        return;
    end
    
    if isfield(s, 'CSASeriesHeaderInfo') && dim(3)>2
        % sSliceArray.asSlice[0].sPosition.dSag/Cor/Tra
        ori = ['Sag'; 'Cor'; 'Tra']; % 1/2/3
        key = sprintf('[%g].sPosition.d%s', ceil(dim(3)/2), ori(iSL,:));
        pos = asc_header(s, ['sSliceArray.asSlice' key]); % mid-slice center loc
        if ~isempty(pos)
            pos = [R(iSL, 1:2) pos] * [-dim(1:2)/2 1]'; % mid-slice location
        end
    end
end

% s.LastFile works for most GE, Philips and all non-mosaic Siemens data
if isempty(pos) && isfield(s, 'LastFile')
    pos = tryGetField(s.LastFile, 'ImagePositionPatient');
    if ~isempty(pos)
        pos = pos(iSL);
        if pos == s.ImagePositionPatient(iSL), pos = []; end % ignore same slice
    end
end

% May be useful for Philips dicom: use volume centre info
if isempty(pos) && isfield(s, 'Stack')
    ori = ['RL'; 'AP'; 'FH']; % x y z
    pos = tryGetField(s.Stack.Item_1, ['MRStackOffcentre' ori(iSL,:)]);
    if ~isempty(pos)
        pos = [R(iSL, 1:2) pos] * [-dim(1:2)/2 1]'; % mid-slice location
    end
end

% GE: s.LastScanLoc is always different from the slice in 1st file
if isempty(pos) && isfield(s, 'LastScanLoc')
    pos = s.LastScanLoc;
    if iSL<3, pos = -pos; end % LastScanLoc uses RAS convention!
end

pos1 = R(iSL, 3:4) * [dim(3)-1 1]'; % last SliceLocation based on current R
if ~isempty(pos) % have SliceLocation for last/center slice
    flip = (pos>R(iSL,4)) ~= (pos1>R(iSL,4)); % same direction?
else % we do some guess work and warn user
    errorLog(['Please check whether slices are flipped: ' ProtocolName(s)]);
    pos2 = R(iSL, 3:4) * [1-dim(3) 1]'; % opposite slice direction
    % if pos1 is larger than the other dir, and is way outside head
    flip = all(abs(pos1) > [abs(pos2) 150]); % arbituary 150 mm
end
if flip, R(:,3) = -R(:,3); end % change to opposite direction

%% Subfunction: return parameter in CSA series ASC header.
function val = asc_header(s, key)
val = []; 
if ~isfield(s, 'CSASeriesHeaderInfo'), return; end
if isfield(s.CSASeriesHeaderInfo, key)
    val = s.CSASeriesHeaderInfo.(key);
    return;
end
if isfield(s.CSASeriesHeaderInfo, 'MrPhoenixProtocol')
    str = s.CSASeriesHeaderInfo.MrPhoenixProtocol;
elseif isfield(s.CSASeriesHeaderInfo, 'MrProtocol') % older version dicom
    str = s.CSASeriesHeaderInfo.MrProtocol;
else
    str = char(s.CSASeriesHeaderInfo');
    k0 = strfind(str, '### ASCCONV BEGIN ###');
    k  = strfind(str, '### ASCCONV END ###');
    str = str(k0:k); % avoid key before BEGIN and after END
end
k = strfind(str, [char(10) key]); % start with new line: safer
if isempty(k), return; end
str = strtok(str(k(1):end), char(10)); % the line
[~, str] = strtok(str, '='); % '=' and the vaule
str = strtrim(strtok(str, '=')); % remvoe '=' and space 

if strncmp(str, '""', 2) % str parameter
    val = str(3:end-2);
elseif strncmp(str, '"', 1) % str parameter for version like 2004A
    val = str(2:end-1);
elseif strncmp(str, '0x', 2) % hex parameter, convert to decimal
    val = sscanf(str(3:end), '%x', 1);
else % decimal
    val = str2double(str);
end

%% Subfunction: return matlab decompress command if the file is compressed.
function func = compress_func(fname)
func = '';
fid = fopen(fname);
if fid<0, return; end
sig = fread(fid, 4, '*uint8')';
fclose(fid);
if isequal(sig, [80 75 3 4]) % zip file
    func = 'unzip';
elseif isequal(sig, [31 139 8 0]) % tgz, tar
    func = 'untar';
end
% ! "c:\program Files (x86)\7-Zip\7z.exe" x -y -oF:\tmp\ F:\zip\3047ZL.zip

%% Subfunction: simplified from Jimmy Shen's NIfTI toolbox
function save_nii(nii, fileprefix)
%  Check file extension. If .gz, unpack it into temp folder
if length(fileprefix) > 2 && strcmp(fileprefix(end-2:end), '.gz')
    if ~strcmp(fileprefix(end-6:end), '.img.gz') && ...
            ~strcmp(fileprefix(end-6:end), '.hdr.gz') && ...
            ~strcmp(fileprefix(end-6:end), '.nii.gz')
        error('Please check filename.');
    end
    gzFile = 1;
    fileprefix = fileprefix(1:end-3);
end

filetype = 1;

%  Note: fileprefix is actually the filename you want to save
if findstr('.nii',fileprefix) & strcmp(fileprefix(end-3:end), '.nii') %#ok<*AND2,*FSTR>
    filetype = 2;
    fileprefix(end-3:end)='';
end

if findstr('.hdr',fileprefix) & strcmp(fileprefix(end-3:end), '.hdr')
    fileprefix(end-3:end)='';
end

if findstr('.img',fileprefix) & strcmp(fileprefix(end-3:end), '.img')
    fileprefix(end-3:end)='';
end

% Xiangrui added this to fix path problem
if ~ispc && strcmp(fileprefix(1:2), '~/')
    fileprefix = [getenv('HOME') fileprefix(2:end)];
end

write_nii(nii, filetype, fileprefix);

%  gzip output file if requested
if exist('gzFile', 'var') && gzFile
    if filetype == 1
        gzipOS([fileprefix, '.img']);
        gzipOS([fileprefix, '.hdr']);
    elseif filetype == 2
        gzipOS([fileprefix, '.nii']);
    end
end

if filetype == 1
    %  So earlier versions of SPM can also open it with correct originator
    M=[[diag(nii.hdr.dime.pixdim(2:4)) -[nii.hdr.hist.originator(1:3).* ...
        nii.hdr.dime.pixdim(2:4)]'];[0 0 0 1]]; %#ok
    save([fileprefix '.mat'], 'M');
end
return					% save_nii

%% Subfunction: use system gzip if available (faster)
function gzipOS(fname)
persistent cmd; % command to run gzip
while isempty(cmd)
    [err, ~] = system('pigz -h'); % pigz on system path?
    if ~err, cmd = 'pigz -f '; break; end
    cmd = fullfile(fileparts(which(mfilename)), 'pigz ');
    [err, ~] = system([cmd '-h']); 
    if ~err, cmd = [cmd '-f ']; break; end
    helpdlg(['No pigz is found on the OS command path. If you have it, '...
        'please add it to your OS path. If not, you are strongly ' ...
        'suggested to install it. It is much faster for gz compression. ' ... 
        'You can install pigz and either add it to the OS path, or simply put ' ...
        'pigz excutable into the same folder as dicm2nii.m.'], 'About pigz');
    [err, ~] = system('gzip -h'); % gzip on system path?
    if ~err, cmd = 'gzip -f '; break; end
    if isempty(which('gzip')) || ~usejava('jvm')
        errorLog(['None of system pigz, gzip or Matlab gzip exists.\n' ...
            'Files won''t be compressed into gz.\n']);
        cmd = false;
        break;
    end
    cmd = true; % use matlab gzip, which is slow    
end

if islogical(cmd)
    if cmd, gzip(fname); delete(fname); end
    return;
end
% eval(['!start "" /B ' cmd '"' fname '"']); % windows background
[err, str] = system([cmd '"' fname '"']); % overwrite if exist
if err, errorLog(['Error during compression:\n' str]); end
return; % gzipOS

%% Subfunction: simplified from Jimmy Shen's NIfTI toolbox
function write_nii(nii, filetype, fileprefix)
hdr = nii.hdr;

switch double(hdr.dime.datatype),
    case   1,    hdr.dime.bitpix = int16(1 );    precision = 'ubit1';
    case   2,    hdr.dime.bitpix = int16(8 );    precision = 'uint8';
    case   4,    hdr.dime.bitpix = int16(16);    precision = 'int16';
    case   8,    hdr.dime.bitpix = int16(32);    precision = 'int32';
    case  16,    hdr.dime.bitpix = int16(32);    precision = 'float32';
    case  32,    hdr.dime.bitpix = int16(64);    precision = 'float32';
    case  64,    hdr.dime.bitpix = int16(64);    precision = 'float64';
    case 128,    hdr.dime.bitpix = int16(24);    precision = 'uint8';
    case 256,    hdr.dime.bitpix = int16(8 );    precision = 'int8';
    case 511,    hdr.dime.bitpix = int16(96);    precision = 'float32';
    case 512,    hdr.dime.bitpix = int16(16);    precision = 'uint16';
    case 768,    hdr.dime.bitpix = int16(32);    precision = 'uint32';
    case 1024,   hdr.dime.bitpix = int16(64);    precision = 'int64';
    case 1280,   hdr.dime.bitpix = int16(64);    precision = 'uint64';
    case 1792,   hdr.dime.bitpix = int16(128);   precision = 'float64';
    otherwise
        error('This datatype is not supported');
end

hdr.dime.glmax = round(double(max(nii.img(:))));
hdr.dime.glmin = round(double(min(nii.img(:))));

ext = [];
esize_total = 0;
if isfield(nii, 'ext') && isstruct(nii.ext) && isfield(nii.ext, 'section')
    ext = nii.ext;
    if ~isfield(ext, 'num_ext')
        ext.num_ext = length(ext.section);
    end
    if ~isfield(ext, 'extension')
        ext.extension = [1 0 0 0];
    end
    
    for i = 1:ext.num_ext
        if ~isfield(ext.section(i), 'ecode') || ~isfield(ext.section(i), 'edata')
            error('Incorrect NIFTI header extension structure.');
        end
        
        n0 = length(ext.section(i).edata) + 8;
        n1 = ceil(n0/16)*16;
        ext.section(i).esize = n1;
        ext.section(i).edata(end+(1:n1-n0)) = zeros(1, n1-n0);
        esize_total = esize_total + n1;
    end
end

if filetype == 2
    fid = fopen([fileprefix '.nii'], 'w');
    if fid < 0, error('Cannot open file %s.nii.',fileprefix); end
    hdr.dime.vox_offset = 352 + esize_total;
    hdr.hist.magic = 'n+1';
    save_nii_hdr(hdr, fid);
    if ~isempty(ext)
        save_nii_ext(ext, fid);
    else
        fwrite(fid, zeros(1, 4), 'uint8');        
    end
else
    fid = fopen([fileprefix '.hdr'], 'w');
    if fid < 0, error('Cannot open file %s.hdr.', fileprefix); end
    hdr.dime.vox_offset = 0;
    hdr.hist.magic = 'ni1';
    save_nii_hdr(hdr, fid);
    if ~isempty(ext)
        save_nii_ext(ext, fid);
    end       
    fclose(fid);

    fid = fopen([fileprefix '.img'], 'w');
end

fwrite(fid, nii.img, precision);
fclose(fid);
return;					% write_nii

%% Subfunction: simplified from Jimmy Shen's NIfTI toolbox
function save_nii_ext(ext, fid)
fwrite(fid, ext.extension, 'uchar');
for i=1:ext.num_ext
    fwrite(fid, ext.section(i).esize, 'int32');
    fwrite(fid, ext.section(i).ecode, 'int32');
    fwrite(fid, ext.section(i).edata, 'uchar');
end

%% Subfunction: simplified from Jimmy Shen's NIfTI toolbox
function save_nii_hdr(hdr, fid)
fseek(fid, 0, -1);
fwrite(fid, hdr.hk.sizeof_hdr(1),    'int32');	% must be 348.
fwrite(fid, padChar(hdr.hk.data_type, 10), 'uchar');
fwrite(fid, padChar(hdr.hk.db_name, 18), 'uchar');
fwrite(fid, hdr.hk.extents(1),       'int32');
fwrite(fid, hdr.hk.session_error(1), 'int16');
fwrite(fid, hdr.hk.regular(1),       'uchar');	% might be uint8
fwrite(fid, hdr.hk.dim_info(1),      'uchar');

fwrite(fid, hdr.dime.dim(1:8),        'int16');
fwrite(fid, hdr.dime.intent_p1(1),  'float32');
fwrite(fid, hdr.dime.intent_p2(1),  'float32');
fwrite(fid, hdr.dime.intent_p3(1),  'float32');
fwrite(fid, hdr.dime.intent_code(1),  'int16');
fwrite(fid, hdr.dime.datatype(1),     'int16');
fwrite(fid, hdr.dime.bitpix(1),       'int16');
fwrite(fid, hdr.dime.slice_start(1),  'int16');
fwrite(fid, hdr.dime.pixdim(1:8),   'float32');
fwrite(fid, hdr.dime.vox_offset(1), 'float32');
fwrite(fid, hdr.dime.scl_slope(1),  'float32');
fwrite(fid, hdr.dime.scl_inter(1),  'float32');
fwrite(fid, hdr.dime.slice_end(1),    'int16');
fwrite(fid, hdr.dime.slice_code(1),   'uchar');
fwrite(fid, hdr.dime.xyzt_units(1),   'uchar');
fwrite(fid, hdr.dime.cal_max(1),    'float32');
fwrite(fid, hdr.dime.cal_min(1),    'float32');
fwrite(fid, hdr.dime.slice_duration(1), 'float32');
fwrite(fid, hdr.dime.toffset(1),    'float32');
fwrite(fid, hdr.dime.glmax(1),        'int32');
fwrite(fid, hdr.dime.glmin(1),        'int32');

fwrite(fid, padChar(hdr.hist.descrip, 80), 'uchar');
fwrite(fid, padChar(hdr.hist.aux_file, 24), 'uchar');
fwrite(fid, hdr.hist.qform_code,    'int16');
fwrite(fid, hdr.hist.sform_code,    'int16');
fwrite(fid, hdr.hist.quatern_b,   'float32');
fwrite(fid, hdr.hist.quatern_c,   'float32');
fwrite(fid, hdr.hist.quatern_d,   'float32');
fwrite(fid, hdr.hist.qoffset_x,   'float32');
fwrite(fid, hdr.hist.qoffset_y,   'float32');
fwrite(fid, hdr.hist.qoffset_z,   'float32');
fwrite(fid, hdr.hist.srow_x(1:4), 'float32');
fwrite(fid, hdr.hist.srow_y(1:4), 'float32');
fwrite(fid, hdr.hist.srow_z(1:4), 'float32');
fwrite(fid, padChar(hdr.hist.intent_name, 16), 'uchar');
fwrite(fid, padChar(hdr.hist.magic, 4), 'uchar');

if ~isequal(ftell(fid), 348), warning('Header size is not 348 bytes.'); end
return;  % save_nii_hdr

%% Subfunction: pad or chop char to correct length. Called by save_nii_hdr
function buf = padChar(ch, len)
len1 = length(ch);
if len1 >= len,  buf = ch(1:len);
else buf = [ch zeros(1, len-len1, 'uint8')];
end

%% Subfunction: simplified from Jimmy Shen's NIfTI toolbox
function nii = make_nii(img)
switch class(img)
    case 'uint8',       datatype = 2;
    case 'int16',       datatype = 4;
    case 'int32',       datatype = 8;
    case 'single',
        if isreal(img), datatype = 16;
        else            datatype = 32;
        end
    case 'double',
        if isreal(img), datatype = 64;
        else            datatype = 1792;
        end
    case 'int8',        datatype = 256;
    case 'uint16',      datatype = 512;
    case 'uint32',      datatype = 768;
    otherwise
        error('Datatype is not supported by make_nii.');
end

dims = size(img);
dims = [length(dims) dims ones(1,8)];
dims = dims(1:8);

hdr.hk.sizeof_hdr       = 348;			% must be 348!
hdr.hk.data_type        = '';
hdr.hk.db_name          = '';
hdr.hk.extents          = 0;
hdr.hk.session_error    = 0;
hdr.hk.regular          = 'r';
hdr.hk.dim_info         = 0;

hdr.dime.dim = dims;
hdr.dime.intent_p1 = 0;
hdr.dime.intent_p2 = 0;
hdr.dime.intent_p3 = 0;
hdr.dime.intent_code = 0;
hdr.dime.datatype = datatype;

mx = round(double(max(img(:))));
mn = round(double(min(img(:))));
switch datatype
    case 2,     img =  uint8(img);  bitpix = 8;
    case 4,     img =  int16(img);  bitpix = 16;
    case 8,     img =  int32(img);  bitpix = 32;
    case 16,    img = single(img);  bitpix = 32;
    case 32,    img = single(img);  bitpix = 64;
    case 64,    img = double(img);  bitpix = 64;
    case 128,   img =  uint8(img);  bitpix = 24;
    case 256,   img =   int8(img);  bitpix = 8;
    case 511,
        img = double(img);
        img = (img - mn)/(mx - mn);
        img = single(img);          bitpix = 96;
        mx = 1;
        mn = 0;
    case 512,   img = uint16(img);  bitpix = 16;
    case 768,   img = uint32(img);  bitpix = 32;
    case 1792,  img = double(img);  bitpix = 128;
    otherwise
        error('Datatype is not supported by make_nii.');
end

hdr.dime.bitpix = bitpix;
hdr.dime.slice_start = 0;
hdr.dime.pixdim = [0 ones(1,7)];
hdr.dime.vox_offset = 0;
hdr.dime.scl_slope = 0;
hdr.dime.scl_inter = 0;
hdr.dime.slice_end = 0;
hdr.dime.slice_code = 0;
hdr.dime.xyzt_units = 0;
hdr.dime.cal_max = 0;
hdr.dime.cal_min = 0;
hdr.dime.slice_duration = 0;
hdr.dime.toffset = 0;
hdr.dime.glmax = mx;
hdr.dime.glmin = mn;

hdr.hist.descrip = '';
hdr.hist.aux_file = 'none';
hdr.hist.qform_code = 0;
hdr.hist.sform_code = 0;
hdr.hist.quatern_b = 0;
hdr.hist.quatern_c = 0;
hdr.hist.quatern_d = 0;
hdr.hist.qoffset_x = 0;
hdr.hist.qoffset_y = 0;
hdr.hist.qoffset_z = 0;
hdr.hist.srow_x = zeros(1,4);
hdr.hist.srow_y = zeros(1,4);
hdr.hist.srow_z = zeros(1,4);
hdr.hist.intent_name = '';
hdr.hist.magic = '';
hdr.hist.originator = zeros(1,5);

nii.hdr = hdr;
nii.img = img;
return;

%% Subfuction: for GUI subfunctions
function dicm2nii_gui(cmd)
hs = guidata(gcbo);
drawnow;
switch cmd
    case 'do_convert'
        if get(hs.srcType, 'Value') > 2 % dicom, PAR, HEAD files
            src = get(hs.src, 'UserData');
        else
            src = get(hs.src, 'String');
        end
        dst = get(hs.dst, 'String');
        if isempty(src) || isempty(dst)
            str = 'Dicom source and Result folder must be specified';
            errordlg(str, 'Error Dialog');
            return;
        end
        rstFmt = get(hs.rstFmt, 'Value') - 1;
        mocoOpt = get(hs.mocoOpt, 'Value') - 1;
        subjName = strtrim(get(hs.subjName, 'String'));
        if get(hs.subjPop, 'Value')==1, subjName = ''; end
        set(hs.convert, 'Enable', 'off', 'string', 'Conversion in progress');
        drawnow;
        try
            dicm2nii(src, dst, rstFmt, mocoOpt, subjName);
        catch me
            try set(hs.convert, 'Enable', 'on', 'String', 'Start conversion'); end
            commandwindow;
            rethrow(me);
        end
        try %#ok<*TRYNC>
            set(hs.convert, 'Enable', 'on', 'String', 'Start conversion');
        end
    case 'popSource'
        i = get(hs.srcType, 'Value');
        txt = {'Dicom folder' 'Zip / tar file' 'Dicom files' 'PAR files' 'HEAD files'};
        set(hs.srcTxt, 'String' , txt{i});
        srcdir = isdir(get(hs.src, 'String'));
        if (i==1 && ~srcdir) || (i>1 && srcdir)
            set(hs.src, 'String', '');
        end
    case 'dstDialog'
        folder = get(hs.dst, 'String'); % current folder
        if ~isdir(folder), folder = get(hs.src, 'String'); end
        if ~isdir(folder), folder = fileparts(folder); end
        if ~isdir(folder), folder = pwd; end
        dst = uigetdir(folder, 'Select a folder to save data files');
        if isnumeric(dst), return; end
        set(hs.dst, 'String' , dst);
    case 'srcDialog'
        folder = get(hs.src, 'String'); % initial folder
        if ~isdir(folder), folder = fileparts(folder); end
        if ~isdir(folder), folder = pwd; end
        i = get(hs.srcType, 'Value');
        if i == 1 % folder
            src = uigetdir(folder, 'Select a folder containing DICOM files');
            if isnumeric(src), return; end
            set(hs.src, 'UserData', src);
        elseif i == 2 % zip/tgz file
            [src, folder] = uigetfile([folder '/*.zip;*.tgz;*.tar;*.tar.gz'], ...
                'Select the compressed file containing data files');
            if isnumeric(src), return; end
            src = fullfile(folder, src);
            set(hs.src, 'UserData', src);
        elseif i == 3 % dicom files
            [src, folder] = uigetfile([folder '/*.dcm'], ...
                'Select one or more DICOM files', 'MultiSelect', 'on');
            if isnumeric(src), return; end
            src = cellstr(src); % in case only 1 file selected
            src = strcat(folder, filesep, src);
            set(hs.src, 'UserData', src);
            src = src{1};
        elseif i == 4 % PAR/REC
            [src, folder] = uigetfile([folder '/*.PAR'], ...
                'Select one or more PAR files', 'MultiSelect', 'on');
            if isnumeric(src), return; end
            src = cellstr(src); % in case only 1 file selected
            src = strcat(folder, src);
            set(hs.src, 'UserData', src);
            src = src{1};
        elseif i == 5 % HEAD/BRIK
            [src, folder] = uigetfile([folder '/*.HEAD'], ...
                'Select one or more HEAD files', 'MultiSelect', 'on');
            if isnumeric(src), return; end
            src = cellstr(src); % in case only 1 file selected
            src = strcat(folder, src);
            set(hs.src, 'UserData', src);
            src = src{1};
        end
        set(hs.src, 'String' , src);
        dicm2nii_gui('pop_subj');
    case 'set_src'
        str = get(hs.src, 'String');
        if ~exist(str, 'file')
            val = dir(str);
            folder = fileparts(str);
            if isempty(val)
                val = get(hs.src, 'UserData');
                if iscellstr(val), val = val{1}; end
                set(hs.src, 'String', val);
                errordlg('Invalid input', 'Error Dialog');
                return;
            end
            str = {val.name};
            str = strcat(folder, filesep, str);
        end
        set(hs.src, 'UserData', str);
        dicm2nii_gui('pop_subj');
    case 'set_dst'
        str = get(hs.dst, 'String');
        if isempty(str), return; end
        if ~exist(str, 'file') && ~mkdir(str)
            set(hs.dst, 'String', '');
            errordlg(['Invalid folder name ''' str ''''], 'Error Dialog');
            return;
        end
    case 'pop_subj'
        if get(hs.subjPop, 'Value')==2 && ...
                ~isempty(strtrim(get(hs.subjName, 'String')))
            return;
        end
        src = get(hs.src, 'UserData');
        if isempty(src), return; end
        if iscellstr(src)
            fname = src;
        elseif isdir(src)
            fname = dir(src);
            fname([fname.isdir]) = [];
            fname = strcat(src, filesep, {fname.name});
        else
            fname = cellstr(src);
        end
        dict = dicm_dict('', {'PatientName' 'PatientID'});
        for i = 1:length(fname)
            s = dicm_hdr(fname{i}, dict);
            if isempty(s), continue; end
            subj = tryGetField(s, 'PatientName');
            if isempty(subj)
                subj = tryGetField(s, 'PatientID', 'unknown'); 
            end
            set(hs.subjName, 'String', subj);
            break;
        end
    otherwise
        create_gui;
end

%% Subfuction: create GUI or bring it to front if exists
function create_gui
set(0,'DefaultFigureWindowStyle','normal')
fh = figure; % arbitury integer
if strcmp('dicm2nii_fig', get(fh, 'Tag')), return; end
scrSz = get(0, 'ScreenSize');
set(fh, 'Toolbar', 'none', 'Menubar', 'none', 'Resize', 'off', ...
    'Tag', 'dicm2nii_fig', 'Position', [200 scrSz(4)-500 420 300], ...
    'Name', 'DICOM to NIfTI Converter', 'NumberTitle', 'off');
clr = get(fh, 'color');

str = 'Choose what kind of data source you are using';
uicontrol('Style', 'text', 'Position', [10 250 90 30], ...
    'FontSize', 9, 'HorizontalAlignment', 'left', ...
    'String', 'Source type', 'Background', clr, 'TooltipString', str);
uicontrol('Style', 'popup', 'Background', 'white', 'Tag', 'srcType', ...
    'String', [' Folder containing dicom files and/or folders|' ...
               ' Compressed file containing data|' ...
               ' Dicom files|' ...
               ' Philips PAR/REC files|' ...
               ' AFNI HEAD/BRIK files'],...
    'Position', [88 254 320 30], 'TooltipString', str, ...
    'Callback', 'dicm2nii(''dicm2nii_gui_cb'',''popSource'');');

str = 'Enter or browse the data source according to the source type';
uicontrol('Style', 'text', 'Position', [10 210 90 30], ...
    'Tag', 'srcTxt', 'FontSize', 9, 'HorizontalAlignment', 'left', ...
    'String', 'Dicom folder', 'Background', clr, 'TooltipString', str);
uicontrol('Style', 'edit', 'Position', [88 220 296 24],'FontSize', 9, ...
    'HorizontalAlignment', 'left', 'Background', 'white', 'Tag', 'src', ...
    'TooltipString', str, ...
    'Callback', 'dicm2nii(''dicm2nii_gui_cb'',''set_src'');');
uicontrol('Style', 'pushbutton', 'Position', [384 221 24 22], ...
    'Tag', 'browseSrc', 'FontSize', 9, 'String', '...', ...
    'TooltipString', 'Browse dicom source', ...
    'Callback', 'dicm2nii(''dicm2nii_gui_cb'',''srcDialog'');');

str = 'Enter or browse a folder to save result files';
uicontrol('Style', 'text', 'Position', [10 170 90 30], ...
    'FontSize', 9, 'HorizontalAlignment', 'left', ...
    'String', 'Result folder', 'Background', clr, 'TooltipString', str);
uicontrol('Style', 'edit', 'Position', [88 180 296 24], 'FontSize', 9, ...
    'HorizontalAlignment', 'left', 'Background', 'white', ...
    'Tag', 'dst', 'TooltipString', str, ...
    'Callback', 'dicm2nii(''dicm2nii_gui_cb'',''set_dst'');');
uicontrol('Style', 'pushbutton', 'Position', [384 181 24 22], ...
    'FontSize', 9, 'String', '...', 'Tag', 'browseDst', ...
    'TooltipString', 'Browse result folder', ...
    'Callback', 'dicm2nii(''dicm2nii_gui_cb'',''dstDialog'');');

str = 'Choose output file format';
uicontrol('Style', 'text', 'Position', [10 130 90 30], 'FontSize', 9, ...
    'HorizontalAlignment', 'left', 'String', 'Output format', ...
    'Background', clr, 'TooltipString', str);
uicontrol('Style', 'popup', 'Background', 'white', 'Tag', 'rstFmt', ...
    'Value', 2, 'Position', [88 134 320 30], 'TooltipString', str, ...
    'String', [' Uncompressed single file (.nii)|' ...
               ' Compressed single file (.nii.gz)|' ...
               ' Uncompressed multiple files (.img / .hdr)|' ...
               ' Compressed multiple files (.img.gz / .hdr.gz)']);

str = 'Choose the way to deal with MoCo series';
uicontrol('Style', 'text', 'Position', [10 90 90 30], 'FontSize', 9, ...
    'HorizontalAlignment', 'left', 'String', 'MoCoSeries', ...
    'Background', clr, 'TooltipString', str);
uicontrol('Style', 'popup', 'Background', 'white', 'Tag', 'mocoOpt', ...
     'Position', [88 94 320 30], 'Value', 2, 'Position', [88 94 320 30], ...
     'TooltipString', str, ...
     'String', [' Use both original and MoCo series|' ...
                ' Ignore MoCo series if both present|' ...
                ' Ignore original series if both present']);

str = ['Enter subject ID in format of Smith^John or Smith (if empty ' ...
       'first name) only if the data contains more than one subjects'];
uicontrol('Style', 'popup', 'Background', clr, 'Tag', 'subjPop', ...
    'String', ' Source is for a single subject| Convert only for subject', ...
    'Position', [10 50 210 30], ...
    'TooltipString', 'Select most likey subject information', ...
    'Callback', 'dicm2nii(''dicm2nii_gui_cb'',''pop_subj'');');
uicontrol('Style', 'edit', 'Position', [224 57 184 24], 'FontSize', 9, ...
    'HorizontalAlignment', 'left', 'Background', 'white', ...
    'Tag', 'subjName', 'TooltipString', str);

uicontrol('Style', 'pushbutton', 'Position', [104 10 200 30], ...
    'FontSize', 9, 'String', 'Start conversion', 'Tag', 'convert', ...
    'TooltipString', 'Dicom source and Result folder needed before start', ...
    'Callback', 'dicm2nii(''dicm2nii_gui_cb'',''do_convert'');');

hs = guihandles(fh); % get handles
guidata(fh, hs); % store handles
set(fh, 'HandleVisibility', 'callback'); % protect from command line
return;

%% subfunction: return phase positive in RAS and iPhase in image space
function [phPos, iPhase] = phaseDirection(s)
ixyz = xform_mat(s);
iPhase = 2;
foo = tryGetField(s, 'InPlanePhaseEncodingDirection', ''); % no for Philips
if strcmp(foo, 'ROW'), iPhase = 1; end
iPhase = ixyz(iPhase);

phPos = [];
if strncmpi(s.Manufacturer, 'SIEMENS', 7)
    phPos = csa_header(s, 'PhaseEncodingDirectionPositive'); % 0 or 1
elseif strncmpi(s.Manufacturer, 'GE', 2)
    fld = 'ProtocolDataBlock';
    if isfield(s, fld) && isfield(s.(fld), 'VIEWORDER')
        phPos = s.(fld).VIEWORDER == 1; % 1 == bottom_up ?
    end
elseif strncmpi(s.Manufacturer, 'Philips', 7) % no InPlanePhaseEncodingDirection
    fld = 'MRStackPreparationDirection';
    if isfield(s, 'Stack') && isfield(s.Stack.Item_1, fld)
        d = s.Stack.Item_1.(fld);
        iPhase = strfind('LRAPSIFH', d(1));
        iPhase = ceil(iPhase/2); 
        if iPhase>3, iPhase = 3; end % 1/2/3 for LR AP FH
        if any(d(1) == 'LPHS'), phPos = false;
        elseif any(d(1) == 'RAFI'), phPos = true;
        end
    end
end
if iPhase ~= 3, phPos = ~phPos; end % dicom LPS vs nii RAS

%% subfunction: extract useful fields for multiframe dicom
function s = multiFrameFields(s)
if any(~isfield(s, {'SharedFunctionalGroupsSequence' ...
        'PerFrameFunctionalGroupsSequence'}))
    return; % do nothing
end
s1 = s.SharedFunctionalGroupsSequence.Item_1;
s2 = s.PerFrameFunctionalGroupsSequence.Item_1;

fld = 'EffectiveEchoTime'; n1 = 'MREchoSequence'; val = [];
if isfield(s1, n1)
    a = s1.(n1).Item_1; val = tryGetField(a, fld);
elseif isfield(s2, n1)
    a = s2.(n1).Item_1; val = tryGetField(a, fld);
end
if ~isfield(s, 'EchoTime') && ~isempty(val), s.EchoTime = val; end
if ~isfield(s, 'EchoTime') && isfield(s, 'EchoTimeDisplay')
	s.EchoTime = s.EchoTimeDisplay;
end

n1 = 'MRTimingAndRelatedParametersSequence';
fld = 'RepetitionTime'; val = [];
if isfield(s1, n1)
    a = s1.(n1).Item_1;  val = tryGetField(a, fld);
elseif isfield(s2, n1)
    a = s2.(n1).Item_1;  val = tryGetField(a, fld);
end
if ~isfield(s, fld) && ~isempty(val), s.(fld) = val; end

fld = 'PixelSpacing'; n1 = 'PixelMeasuresSequence'; val = [];
if isfield(s1, n1)
    a = s1.(n1).Item_1;  val = tryGetField(a, fld);
elseif isfield(s2, n1)
    a = s2.(n1).Item_1;  val = tryGetField(a, fld);
end
if ~isfield(s, fld) && ~isempty(val), s.(fld) = val; end

fld = 'SpacingBetweenSlices';  val = [];
if isfield(s1, n1)
    a = s1.(n1).Item_1;  val = tryGetField(a, fld);
elseif isfield(s2, n1)
    a = s2.(n1).Item_1;  val = tryGetField(a, fld);
end
if ~isfield(s, fld) && ~isempty(val), s.(fld) = val; end

fld = 'SliceThickness'; val = [];
if isfield(s1, n1)
    a = s1.(n1).Item_1;  val = tryGetField(a, fld);
elseif isfield(s2, n1)
    a = s2.(n1).Item_1;  val = tryGetField(a, fld);
end
if ~isfield(s, fld) && ~isempty(val), s.(fld) = val; end

fld = 'RescaleIntercept'; n1 = 'PixelValueTransformationSequence'; val = [];
if isfield(s1, n1)
    a = s1.(n1).Item_1;  val = tryGetField(a, fld);
elseif isfield(s2, n1)
    a = s2.(n1).Item_1;  val = tryGetField(a, fld);
end
if ~isfield(s, fld) && ~isempty(val), s.(fld) = val; end

fld = 'RescaleSlope'; val = [];
if isfield(s1, n1)
    a = s1.(n1).Item_1;  val = tryGetField(a, fld);
elseif isfield(s2, n1)
    a = s2.(n1).Item_1;  val = tryGetField(a, fld);
end
if ~isfield(s, fld) && ~isempty(val), s.(fld) = val; end

fld = 'ImageOrientationPatient'; n1 = 'PlaneOrientationSequence'; val = [];
if isfield(s1, n1)
    a = s1.(n1).Item_1;  val = tryGetField(a, fld);
elseif isfield(s2, n1)
    a = s2.(n1).Item_1;  val = tryGetField(a, fld);
end
if ~isfield(s, fld) && ~isempty(val), s.(fld) = val; end

fld = 'ImagePositionPatient'; n1 = 'PlanePositionSequence';
if isfield(s2, n1)
    a = s2.(n1); val = tryGetField(a.Item_1, fld);
    if ~isfield(s, fld) && ~isempty(val), s.(fld) = val; end
end

s2 = s.PerFrameFunctionalGroupsSequence;
n1 = fieldnames(s2);
s2 = s2.(n1{end}); % last frame

% check ImageOrientationPatient consistency for 1st and last frame only
fld = 'ImageOrientationPatient'; n1 = 'PlaneOrientationSequence';
if isfield(s2, n1)
    a = s2.(n1).Item_1; val = tryGetField(a, fld);
    if ~isempty(val)
        try
            if sum(abs(val-s.ImageOrientationPatient))>0.01
                s = rmfield(s, 'ImageOrientationPatient');
                return; % inconsistent orientation, remove the field
            end
        end
    end
end

fld = 'ImagePositionPatient'; n1 = 'PlanePositionSequence';
if isfield(s2, n1)
    a = s2.(n1).Item_1; val = tryGetField(a, fld);
    if ~isempty(val), s.LastFile.(fld) = val; end
end

fld = 'ComplexImageComponent'; n1 = 'MRImageFrameTypeSequence';
if isfield(s2, n1)
    a = s2.(n1).Item_1; val = tryGetField(a, fld);
    if ~isempty(val), s.LastFile.(fld) = val; end
end

fld = 'RescaleIntercept'; n1 = 'PixelValueTransformationSequence';
if isfield(s2, n1)
    a = s2.(n1).Item_1; val = tryGetField(a, fld);
    if ~isempty(val), s.LastFile.(fld) = val; end
end
fld = 'RescaleSlope';
if isfield(s2, n1)
    a = s2.(n1).Item_1; val = tryGetField(a, fld);
    if ~isempty(val), s.LastFile.(fld) = val; end
end

fld = 'ImagePositionPatient'; n1 = 'PlanePositionSequence';
s2 = s.PerFrameFunctionalGroupsSequence.Item_2;
if isfield(s2, n1)
    a = s2.(n1).Item_1; val = tryGetField(a, fld);
    if isfield(s, fld) && sum(abs(s.(fld)-val))<0.01
        s.Dim3IsVolume = true;
    end
end

%% subfunction: split nii into mag and phase for Philips single file
function nii = save_philips_phase(nii, s, dataFolder, fname, ext, fmtStr)
if ~strcmp(tryGetField(s, 'ComplexImageComponent', ''), 'MIXED')
    return; % do nothing if not MIXED image
end

if ~isfield(s, 'VolumeIsPhase') % PAR file has this already
    dim = nii.hdr.dime.dim(4:5);
    if tryGetField(s, 'Dim3IsVolume'), iFrames = 1:dim(2);
    else iFrames = 1:dim(1):dim(1)*dim(2);
    end
    flds = {'PerFrameFunctionalGroupsSequence' ...
        'MRImageFrameTypeSequence' 'ComplexImageComponent'};
    if dim(2) == 2 % 2 volumes, no need to re-read ComplexImageComponent
        iFrames(2) = dim(1)*dim(2); % use last frame
        s1.(flds{1}) = s.(flds{1});        
    else
        dict = dicm_dict(s.Manufacturer, flds);
        s1 = dicm_hdr(s.Filename, dict, iFrames);
    end
    s.VolumeIsPhase = false(dim(2), 1);
    for i = 1:dim(2)
        Item = sprintf('Item_%g', iFrames(i));
        foo = s1.(flds{1}).(Item).(flds{2}).Item_1.(flds{3});
        s.VolumeIsPhase(i) = strcmp(foo, 'PHASE');
    end
end

niiP = nii;
niiP.img = nii.img(:,:,:,s.VolumeIsPhase);
n = sum(s.VolumeIsPhase);
niiP.hdr.dime.dim(5) = n; % may be 1 always
niiP.hdr.dime.dim(1) = 3 + (n>1);

nii.img(:,:,:,s.VolumeIsPhase) = []; % now only mag
n = sum(~s.VolumeIsPhase);
nii.hdr.dime.dim(5) = n; % may be 1 always
nii.hdr.dime.dim(1) = 3 + (n>1);

% undo scale for 2nd set img if it was applied to img in set_nii_header
if (nii.hdr.dime.scl_inter==0) && (nii.hdr.dime.scl_slope==1) && ...
        (tryGetfield(s, 'RescaleIntercept') ~=0 ) && ...
        (tryGetfield(s, 'RescaleSlope') ~= 1)
    if s.VolumeIsPhase(1)
        nii.img = (nii.img - s.RescaleIntercept) / s.RescaleSlope;
        nii.hdr.dime.scl_inter = s.LastFile.RescaleIntercept;
        nii.hdr.dime.scl_slope = s.LastFile.RescaleSlope;
    else
        niiP.img = (niiP.img - s.RescaleIntercept) / s.RescaleSlope;
        niiP.hdr.dime.scl_inter = s.LastFile.RescaleIntercept;
        niiP.hdr.dime.scl_slope = s.LastFile.RescaleSlope;
    end
end

fprintf(fmtStr, [fname '_phase'], 1);
save_nii(niiP, [dataFolder fname '_phase' ext]); % save phase nii

%% Write error info to a file in case user ignores Command Window output
function errorLog(errInfo)
if isempty(errInfo), return; end
fprintf(2, ' %s\n', errInfo); % red text in Command Window
global dcm2nii_errFileName;
fid = fopen(dcm2nii_errFileName, 'a');
fseek(fid, 0, -1); 
fprintf(fid, '%s\n', errInfo);
fclose(fid);
