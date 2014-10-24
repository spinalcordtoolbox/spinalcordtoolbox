function rename_dicm(files, fmt)
% rename_dicm(files, outputNameFormat)
% 
% Rename dicom files so the names will be human readable.
% 
% The first input is the dicom file(s) or a folder containing dicom files.
% The second input is the format for the result file names. Support format
% include:
% 
% 1: Protocol_#####.dcm, such as run1_00001.dcm. If there is MoCo series,
%    or users did not change run names, there will be name conflict.
%   
% 2: Subj-Ser#-Acq#-Inst#.dcm, such as 2334ZL-0004-0001-00001.dcm. This is
%    the BrainVoyager format. It won't have any name confict, but it is
%    long and less descriptive. Note that BrainVoyager itself has problem
%    to distinguish the two series of images for Siemens fieldmap, while
%    this code can avoid this problem.
% 
% 3: Protocol_Se#_Inst#, such as run1_004_00001.dcm. This gives short names,
%    while it is descriptive and there is no name conflict most of time.
% 
% 4: Subj_Protocol_In#, such as 2334ZL_run1_00001.dcm. This is useful if
%    files for different subjects are in the same folder.
% 
% 5: Protocol_Ser#-Acq#-Inst#, such as run1_003_001_00001.dcm. This ensures 
%    no name conflict, and is the default.
% 
% Whenever there is name confict, you will see red warning and the latter
% files won't be renamed.
% 
% If the first input is not provided or empty, you will be asked to pick up
% a folder.
 
% 2007.10   Write it (Xiangrui Li)
% 2013.04   Add more options for output format
% 2013.06   Exclude PhoenixZIPReport files to avoid error
% 2013.06   Fix problem if illegal char in ProtocolName
% 2013.09   Use dicm_hdr to replace dicominfo, so it runs much faster
% 2013.09   Use 5-diget InstanceNumber, so works better for GE/Philips
% 2014.02   Add Manufacturer to flds (bug caused by dicm_hdr update)
% 2014.05   Use SeriesDescription to replace ProtocolName

curFolder = pwd;
clnObj = onCleanup(@() cd(curFolder));
if nargin<1 || isempty(files)
    folder = uigetdir(cd,'Select a folder containing DICOM files');
    if folder==0, return; end
    cd(folder);
    files = dir;
    files([files.isdir]) = [];
    files = {files.name};
    
    str = sprintf(['Choose Output format: \n\n' ...
                   '1: run1_00001.dcm\n' ...
                   '2: BrainVoyager format\n' ...
                   '3: run1_001_00001.dcm\n' ...
                   '4: subj_run1_00001.dcm\n' ...
                   '5: run1_001_001_00001.dcm\n']);
    fmt = inputdlg(str, 'Rename Dicom', 1, {'5'});
    if isempty(fmt), return; end
    fmt = str2double(fmt{1});
else
    if exist(files, 'dir') % input is folder
        cd(files);
        files = dir;
        files([files.isdir]) = [];
        files = {files.name};
    else % files
        if ~iscell(files), files = {files}; end
        folder = fileparts(files{1});
        if ~isempty(folder), cd(folder); end
    end
    if nargin<2 || isempty(fmt), fmt = 5; end
end

if ispc, ren = 'rename';
else ren = 'mv';
end % matlab movefile is too slow

flds = {'InstanceNumber' 'AcquisitionNumber' 'SeriesNumber' 'EchoNumber' 'ProtocolName' ...
        'SeriesDescription' 'PatientName' 'PatientID' 'Manufacturer'};
dict = dicm_dict('', flds);

nFile = length(files);
if nFile<1, return; end
err = '';
str = sprintf('%g/%g', 1, nFile);
fprintf(' Renaming DICOM files: %s', str);

for i = 1:nFile
    fprintf(repmat('\b', [1 length(str)]));
    str = sprintf('%g/%g', i, nFile);
    fprintf('%s', str);
    s = dicm_hdr(files{i}, dict);
    try % skip if no these fields
        sN = s.SeriesNumber;
        aN = s.AcquisitionNumber;
        iN = s.InstanceNumber;
        if strncmp(s.Manufacturer, 'SIEMENS', 7)
            pName = strtrim(s.ProtocolName);
        else
            pName = strtrim(s.SeriesDescription);
        end
        if isfield(s, 'PatientName')
            sName = s.PatientName;
        else
            sName = s.PatientID;
        end
    catch me %#ok
        continue;
    end
    
    pName(~isstrprop(pName, 'alphanum')) = '_'; % make str valid for file name
    while 1
        ind = strfind(pName, '__');
        if isempty(ind), break; end
        pName(ind) = [];
    end
    sName(~isstrprop(sName, 'alphanum')) = '_'; % make str valid for file name
    while 1
        ind = strfind(sName, '__');
        if isempty(ind), break; end
        sName(ind) = [];
    end
    
    if strncmpi(s.Manufacturer, 'Philips', 7) % SeriesNumber is useless
        sN = aN;
    elseif strncmpi(s.Manufacturer, 'SIEMENS', 7)
        if isfield(s, 'EchoNumber') && s.EchoNumber>1
            aN = s.EchoNumber;  % fieldmap phase image
        end
    end
    
    if fmt == 1 % pN_001
        name = sprintf('%s_%05g.dcm', pName, iN);
    elseif fmt == 2 % BrainVoyager
        name = sprintf('%s-%04g-%04g-%05g.dcm', sName, sN, aN, iN);
    elseif fmt == 3 % pN_03_00001
        name = sprintf('%s_%02g_%05g.dcm', pName, s.SeriesNumber, iN);
    elseif fmt == 4 % 2322ZL_pN_001
        name = sprintf('%s_%s_%05g.dcm', sName, pName, iN); 
    elseif fmt == 5 % pN_003_001_001
        name = sprintf('%s_%03g_%03g_%05g.dcm', pName, sN, aN, iN); 
    else
        error('Invalid format.');
    end
    
    if strcmpi(files{i}, name), continue; end % done already
    [er, foo] = system([ren ' "' files{i} '" ' name]);
    if er, err = [err files{i} ': ' foo]; end %#ok
end
fprintf('\n');
if ~isempty(err), fprintf(2, '\n%s\n', err); end
