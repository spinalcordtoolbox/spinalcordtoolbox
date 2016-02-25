function [vols,volsizes,inplanematrixsizes,trs] = dicomloaddir(files,filenameformat,maxtoread,desiredinplanesize,phasemode,dformat)

% function [vols,volsizes,inplanematrixsizes,trs] = dicomloaddir(files,filenameformat,maxtoread,desiredinplanesize,phasemode,dformat)
%
% <files> is a pattern matching zero or more DICOM directories.  if things
%   other than directories are matched, we just silently ignore them.
% <filenameformat> (optional) is a string with the format of the DICOM
%   file names (e.g. 'i%06d.dcm').  we assume that the string accepts 
%   a 1-indexed integer.  if <filenameformat> is supplied, we assume that all files in the
%   DICOM directories are of that format.  default is [] which means to 
%   try to match all files, and then perform numerical sorting (see below).  
%   the potential problem with matching all files is that
%   there may be too many files for the ls UNIX command to match.
%   numerical sorting means to extract all sequences of digits from each file name,
%   and then sort those digits using precedence from left to right in order 
%   to determine the file order (this is useful for files named like:
%     'MR.1.2.840.113619.2.283.4120.7575399.14698.1315187063.22').
% <maxtoread> (optional) is a positive integer with the maximum number 
%   of slices to read in.  should be a multiple of the number of slices
%   in a single volume.  default is [] which means to do not impose a limit.
%   (note that if there are multiple versions of each slice, e.g. because you
%   saved out phase data, <maxtoread> should be enlarged appropriately.)
% <desiredinplanesize> (optional) is [A B] with the desired matrix size of each
%   slice.  if [A B] does not match what is read in, we automatically imresize
%   the slice using 'lanczos3' interpolation, <volsizes> is updated
%   accordingly, and the final reporting to the command window is updated
%   accordingly.  if [] or not supplied, we do nothing special.
% <phasemode> (optional) is {D E} where D is an integer describing the 
%   write mode and E is a vector of numbers indicating which versions to output
%   in the rows of <vols>.  the write mode should be a 4-bit integer where 
%   high to low bit indicates imaginary, real, phase, magnitude; these various
%   versions are assumed to be written out in reverse order.  for example, D==13
%   means that for each slice, the magnitude, real, and imaginary parts
%   are written out as separate DICOM files in that order.  each element of E
%   should be 1=magnitude, 2=phase, 3=real, 4=imaginary, 5=magnitude computed
%   from real and imaginary, and 6=phase computed from real and imaginary.
%   in the last case (element of E equal to 6), the range of values will be [-pi,pi].
%   in the other cases, the range of values will be as given by the DICOM files.
%   we will crash if there is an entry in E that cannot be computed based on D.
%   default is {1 1} which corresponds to case where only one version of the
%   data is provided (magnitude images) and you want that version returned.
%   Note that <phasemode> is not supported for Siemens data files.
% <dformat> (optional) is 'single' | 'double' with the format to cast the
%   data to.  default: 'double'.
%
% load the DICOM files in each directory.  return <vols>
% as a cell vector of matrices (format of <dformat>), <volsizes> as
% a cell vector of 3-element vectors with the voxel sizes in mm,
% <inplanematrixsizes> as a cell vector of [A B] indicating
% the frequency-encode and phase-encode matrix dimensions, and
% <trs> as a cell vector of scalars with the repetition time in seconds.
% when <phasemode> is supplied, <vols> may have more than one row
% to accommodate different phase-related versions of the data.
%
% to figure out voxel sizes and in-plane matrix sizes, we look at 
% the DICOM header information of the first slice, specifically, the fields
% 'PixelSpacing' and 'SliceThickness' for voxel sizes and the fields
% 'Private_0027_1060' and 'Private_0027_1061' for the in-plane matrix
% sizes.  to figure out the number of slices in one volume, we examine 
% the field 'Private_0021_104f'.
%
% the data are returned with dimensions X x Y x Z x T, where T could equal 1.
%
% if one or more of the things matched by <files> is not a DICOM
% directory, we just skip gracefully (and report to the command window).
%
% history:
% 2014/04/27 - better handling of file number matching; just issue warning now when in-plane matrix size is not found
% 2011/09/16 - change default behavior so that it matches * and perform numerical sorting for free
% 2011/08/07 - allow zero directories to be matched.
% 2011/07/27 - add trs output
% 2011/07/26 - fix minor bugs (it would have crashed)
% 2011/07/21 - silently ignore non-directories; add support for Siemens data
% 2011/07/15 - add input <dformat>
% 2011/07/15 - do a temporary change of directory to reduce failure of ls 
%              for large directory names and large number of files
% 2011/07/11 - enforce extra assumption when <filenameformat> is supplied, so
%              that we can implement a big speedup.
% 2011/04/05 - add <phasemode>
% 2011/04/03 - add status dots
% 2011/03/29 - add input <desiredinplanesize>

% TODO:
% - report out acquisition time?

% input
if ~exist('filenameformat','var') || isempty(filenameformat)
  filenameformat = [];
end
if ~exist('maxtoread','var') || isempty(maxtoread)
  maxtoread = [];
end
if ~exist('desiredinplanesize','var') || isempty(desiredinplanesize)
  desiredinplanesize = [];
end
if ~exist('phasemode','var') || isempty(phasemode)
  phasemode = {1 1};
end
if ~exist('dformat','var') || isempty(dformat)
  dformat = 'double';
end

% match the pattern
files = matchfiles(files);
%assert(length(files) >= 1,'<files> does not match at least one directory');

% do it
vols = {}; volsizes = {}; inplanematrixsizes = {}; trs = {};
cnt = 0;
for p=1:length(files)
  
  % check if directory
  if ~exist(files{p},'dir')
    continue;
  end

  % match all files
  if isempty(filenameformat)
      tempdir = pwd;
      cd(files{p});
    files0 = matchfiles('*');
      % ok, collect the numbers
    filenumbers = [];
    for qqq=1:length(files0)
      temp = regexp(files0{qqq},'(\d+)','tokens');
      filenumbers(qqq,:) = cellfun(@(x) str2double(x{1}),temp(1));
      if qqq==1
        filenumbers = placematrix(zeros(length(files0),size(filenumbers,2)),filenumbers,[1 1]);
      end
    end
      % sort the numbers and re-order the files
    [d,iii] = sortrows(filenumbers,1:size(filenumbers,2));
    files0 = files0(iii);
      % continue on
    for ppp=1:length(files0)
      files0{ppp} = [files{p} '/' files0{ppp}];
    end
      cd(tempdir);
    if ~isempty(maxtoread)
      files0 = files0(1:min(maxtoread,end));
    end
    if isempty(files0)
      fprintf('this (%s) does not appear to be a directory with DICOM files, so skipping.\n',files{p});
      continue;
    else
      fprintf('this (%s) appears to be a directory with DICOM files, so loading.\n',files{p});
      cnt = cnt + 1;
    end
  else
    numfiles = str2double(unix_wrapper(sprintf('find "%s" | wc -l',files{p}),0)) - 1;
    files0 = {};
    for q=1:numfiles
      if ~isempty(maxtoread) && q > maxtoread
        break;
      end
      files0{q} = [files{p} '/' sprintf(filenameformat,q)];
    end
    if isempty(files0) || ~exist(files0{1},'file')
      fprintf('this (%s) does not appear to be a directory with DICOM files, so skipping.\n',files{p});
      continue;
    else
      fprintf('this (%s) appears to be a directory with DICOM files, so loading.\n',files{p});
      cnt = cnt + 1;
    end

%     % FIXME: the following is not really necessary (and therefore overly slow) since we could just try and catch the dicomread
%     q = 1;
%     files0 = {};
%     while 1
%       if ~isempty(maxtoread) && q > maxtoread
%         break;
%       end
%       filename = [files{p} '/' sprintf(filenameformat,q)];
%       if exist(filename,'file')
%         files0{q} = filename;
%         q = q + 1;
%       else
%         break;
%       end
%     end

  end

  % get some DICOM info
  a = dicominfo(subscript(files0,1,1));
  
  % is this the mosaic case?
  ismosaic = isfield(a,'Private_0051_1016') && ~isempty(regexp(flatten(char(a.Private_0051_1016)),'MOSAIC'));
  
  % figure out number of slices
  if isfield(a,'Private_0021_104f')
    numslices = double(a.Private_0021_104f);  % GE
  elseif isfield(a,'Private_0019_100a')
    numslices = double(a.Private_0019_100a(1));  % Siemens
  else
    numslices = NaN;  % could not find, but that's okay, we'll just concatenate
    assert(~ismosaic);
  end
  
  % what versions do we have (in numerical order)? [mag phase real imag].  e.g.: [1 0 1 1].
  wehave = [bitget(phasemode{1},1) bitget(phasemode{1},2) bitget(phasemode{1},3) bitget(phasemode{1},4)];
  N = sum(wehave);  % total number of versions per slice

  % which versions do we need to actually read in? e.g.: [0 0 1 1] means we need real and imag.
  weneed = [ismember(1,phasemode{2}) ismember(2,phasemode{2}) any(ismember([3 5 6],phasemode{2})) any(ismember([4 5 6],phasemode{2}))];
  weneedf = find(weneed);
  
  % sanity check that we have what we need
  assert(all(~weneed | (weneed & wehave)),'we do not have the image versions requested!');

  % which slice versions (relative to what exists in the directory) should we read in?
  readix = find(ismember(find(wehave),find(weneed)));

  % read in the data
  for q=1:length(files0)/N
    statusdots(q,length(files0)/N);
    
    % fill out a temporary cell vector of length 4.  we fill in only the elements that we actually need.
    temp = {};
    for qq=1:length(readix)
      temp0 = cast(dicomread(files0{(q-1)*N+readix(qq)}),dformat);
      assert(ndims(temp0)<=2);
      
      % if this appears to be a mosaic image, then reshape it to become a normal volume   [MANY ASSUMPTIONS ON DICOM FORMAT HERE!!]
      if ismosaic
        totaldim = sizefull(temp0,2);  % matrix size of total mosaic, e.g. 768 x 480
        mosaicdim = repmat(ceil(sqrt(numslices)),[1 2]);  % gridding of the mosaic, e.g. 8 x 8
        temp0 = subscript(catcell(3,mat2cell(temp0,repmat(totaldim(1)/mosaicdim(1),[1 mosaicdim(1)]),repmat(totaldim(2)/mosaicdim(2),[1 mosaicdim(2)]))'),{':' ':' 1:numslices});  % X x Y x Z
      end
      
      % deal with desiredinplanesize
      if ~isempty(desiredinplanesize) && ~isequal(sizefull(temp0,2),desiredinplanesize)
        rsfactor = sizefull(temp0,2)./desiredinplanesize;
        temp0 = processmulti(@imresize,temp0,desiredinplanesize,'lanczos3');
      end
      
      % record
      temp{weneedf(qq)} = temp0;

    end
    
    % ok, make a cell column vector with the versions we actually want.
    finalvol = {};
    for qq=1:length(phasemode{2})
      switch phasemode{2}(qq)
      case 1
        finalvol{end+1,1} = temp{1};
      case 2
        finalvol{end+1,1} = temp{2};
      case 3
        finalvol{end+1,1} = temp{3};
      case 4
        finalvol{end+1,1} = temp{4};
      case 5
        finalvol{end+1,1} = sqrt(temp{3}.^2 + temp{4}.^2);
      case 6
        finalvol{end+1,1} = atan2(temp{4},temp{3});
      end
    end
    
    % merge it in
    if q==1
      for qq=1:size(finalvol,1)  % qq is index of the version.  cnt is the index of the directory that we're on.
        vols{qq,cnt} = placematrix2(zeros(size(finalvol{qq},1),size(finalvol{qq},2),size(finalvol{qq},3),length(files0)/N,dformat),finalvol{qq});
      end
    else
      for qq=1:size(finalvol,1)
        vols{qq,cnt}(:,:,:,q) = finalvol{qq};
      end
    end
  end
  % now, vols{qq,cnt} should be X x Y x 1 x Z*T (GE) or X x Y x Z x T (Siemens mosaic)
 
  % figure out voxel size
  volsizes{cnt} = double([flatten(a.PixelSpacing) a.SliceThickness]);
  if exist('rsfactor','var')
    volsizes{cnt}(1:2) = volsizes{cnt}(1:2) .* rsfactor;
  end
  
  % figure out in-plane matrix size
  if isfield(a,'Private_0027_1060')
    inplanematrixsizes{cnt} = double([a.Private_0027_1060 a.Private_0027_1061]);  % GE
  elseif isfield(a,'Private_0051_100b')
    tokens = regexp(flatten(char(a.Private_0051_100b)),'(?<pelines>\d+).*\*(?<felines>\d+)','names');
    inplanematrixsizes{cnt} = [str2double(tokens.felines) str2double(tokens.pelines)];  % Siemens
% HRM, NOT SURE ABOUT THIS ANYMORE:
%     if isfield(a,'NumberOfPhaseEncodingSteps')
%       assert(double(a.NumberOfPhaseEncodingSteps) == str2double(tokens.pelines));  % sanity check
%     end
  else
     fprintf('warning, can''t find inplanematrixsizes\n');
     inplanematrixsizes{cnt} = [NaN NaN];
%    die;
  end
  
  % figure out TR (in seconds)
  if isfield(a,'RepetitionTime')
    trs{cnt} = double(a.RepetitionTime)/1000;
  else
    die;
  end

  % if there appear to be multiple temporal samples, reshape the volume [THIS COULD BE DONE EARLIER, ABOVE...]
  for qq=1:size(vols,1)
    if isnan(numslices)
      vols{qq,cnt} = reshape(vols{qq,cnt},size(vols{qq,cnt},1),size(vols{qq,cnt},2),[]);
    else
      vols{qq,cnt} = reshape(vols{qq,cnt},size(vols{qq,cnt},1),size(vols{qq,cnt},2),numslices,[]);
    end
  end

  % report
  fprintf('The 3D dimensions of the final returned volume are %s.\n',mat2str(sizefull(vols{1,cnt},3)));
  if size(vols{1,cnt},4) > 1
    fprintf('There are %s volumes in the fourth dimension.\n',mat2str(size(vols{1,cnt},4)));
  end
  fprintf('The voxel size (mm) of the final returned volume is %s.\n',mat2str(volsizes{cnt}));
  fprintf('The field-of-view (mm) of the final returned volume is %s.\n',mat2str(sizefull(vols{1,cnt},3) .* volsizes{cnt}));
  fprintf('The in-plane matrix size (FE x PE) appears to be %s.\n',mat2str(inplanematrixsizes{cnt}));
  fprintf('The TR is %s seconds.\n',mat2str(trs{cnt}));
  if isfield(a,'Private_0043_102c')
    fprintf('The read-out time per PE line is %s microseconds.\n',mat2str(a.Private_0043_102c));  % GE.  CAN WE FIGURE THIS OUT FOR SIEMENS?
  end
  if isfield(a,'InPlanePhaseEncodingDirection')
    fprintf('The phase-encoding direction is %s.\n',a.InPlanePhaseEncodingDirection);
  end
  if isfield(a,'Private_0019_1093')
    fprintf('The center frequency is %d.\n',a.Private_0019_1093);  % GE.  CAN WE FIGURE THIS OUT FOR SIEMENS?
  end

end







% GE:
%                   NumberOfTemporalPositions: 114
%                         ImagesInAcquisition: 2394
%   PixelBandwidth: 7812.5
%   ReconstructionDiameter: 160
%   episliceorder
% AcquisitionMatrix
