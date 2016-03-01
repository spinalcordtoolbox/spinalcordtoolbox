function [vols,paramsB] = motioncorrectvolumes(vols,volsize,figuredir,ref,cutoff,extratrans,skipreslice, ...
  realignparams,resliceparams,binarymask,epiignoremcvol,dformat)

% function [vols,paramsB] = motioncorrectvolumes(vols,volsize,figuredir,ref,cutoff,extratrans,skipreslice, ...
%   realignparams,resliceparams,binarymask,epiignoremcvol,dformat)
%
% <vols> is
%   (1) a time-series of 3D volumes, X x Y x Z x T.  the volumes should be
%       suitable for interpretation as int16.
%   (2) a cell vector of one or more things like (1).  the first three
%       dimensions (X, Y, Z) must be the same across the various cases.
% <volsize> is [a b c d] where a, b, and c are lengths in millimeters
%   corresponding to dimensions X, Y, and Z, and d is a time in seconds
%   corresponding to dimension T.  for example, [3 3 3 1] indicates isotropic
%   3-mm voxels that are measured every 1 second.  can also be a cell vector
%   of things like [a b c d]; however, the sizes (a, b, c) must be the same
%   across the various cases.
% <figuredir> (optional) is the directory to write figures to.  default is [] which
%   means to not write figures.
% <ref> (optional) indicates which 3D volume should be used as the reference.  the
%   format is [g h] where g indicates the index of the time-series and h indicates
%   the index of the volume within that time-series.  for example, [2 1] indicates
%   that the first volume in the second time-series should be used as the reference.
%   can also be the 3D volume itself (we use this case if ~isvector(<ref>)).
%   default: [1 1].
% <cutoff> (optional) controls the filtering applied to motion parameter estimates.
%   A means low-pass filter cutoff in Hz.  can be Inf.
%  -B means high-pass filter cutoff in Hz.
%  [A B] means band-pass filter cutoffs in Hz.
%  default: 1/90.
% <extratrans> (optional) is a 4x4 transformation matrix that maps points in the
%   matrix space of the reference volume to a new location.  if supplied,
%   then volumes will be resampled at the new location.  for example, if <extratrans>
%   is [1 0 0 1; 0 1 0 0; 0 0 1 0; 0 0 0 1], then this will cause volumes to be 
%   resampled at a location corresponding to a one-voxel shift along the first
%   dimension.  note that we interpolate only once in the entire process.
%   default: eye(4).
% <skipreslice> (optional) is whether to skip the re-slicing procedure.
%   in this case, <vols> remains unchanged in the output and <extratrans> has
%   no effect.  default: 0.
% <realignparams> (optional) is a struct of parameters for spm_realign.m.
%   default: struct('quality',1,'fwhm',3,'sep',2,'interp',7).
% <resliceparams> (optional) is a struct of parameters for spm_reslice.m.
%   default: struct('mask',1,'mean',0,'interp',7,'which',2).
% <binarymask> (optional) is X x Y x Z with 0/1 indicating which voxels to use
%   in the motion parameter estimation.  default is [] which means to do 
%   nothing special.
% <epiignoremcvol> (optional) is a vector of indices indicating which volumes
%   whose motion parameter estimates to ignore.  can be a cell vector of things
%   like that, one for each time-series in <vols>.  default is [] which means
%   to do nothing special.  if supplied, we will ignore the motion parameter
%   estimates for the volumes indicated and use the estimates obtained for the
%   volume(s) that are closest in the temporal sense (see nantoclosest.m).
%   note that this comes before temporal filtering.  and note that the figures
%   that are written reflect the motion parameter estimates after handling
%   <epiignoremcvol>.
% <dformat> (optional) is the datatype to cast the results to.  default: 'double'.
%
% coregister each time-series of volumes to the <ref> volume.
% return the resliced volumes in <dformat> format in <vols>.
% the dimensions will be exactly the same as what was passed in for <vols>.
% also, return <paramsB> which is a cell vector of matrices that are N+1 x 12.
%   (the N can differ across cases.)  each matrix has the filtered motion 
%   parameter estimates.
%
% we perform coregistration by using routines from SPM (e.g. spm_realign).  note that for 
% each time-series of volumes, voxels that move outside of the field-of-view will have 0s
% in all volumes from that time-series.  (thus, the locations of the 0s may differ across
% different time-series.)
%
% note that we use some defaults for the SPM function calls (see code).
% also we assume SPM writes out 'int16' files.
%
% we have tested only SPM5.  no guarantees on whether this works for other versions of SPM.
%
% to speed things up, we attempt to use parfor to process each time-series in <vols> in parallel.
%
% see also coregistervolumes.m.
%
% history:
% 2011/04/13 - add input <epiignoremcvol>
% 2011/03/25 - add input <binarymask>
% 2011/03/15 - add inputs <realignparams> and <resliceparams>
% 2010/03/03 - introduce parfor here.
%
% example:
% vol = getsamplebrain(1)*30000;
% volB = undistortvolumes(vol,[2.5 2.5 2.5],rand(64,64,19)*8,2,[]);
% newvols = motioncorrectvolumes(cat(4,vol,volB),[2.5 2.5 2.5 1],[],[],Inf);
% figure; imagesc(makeimagestack(vol));
% figure; imagesc(makeimagestack(volB));
% figure; imagesc(makeimagestack(newvols(:,:,:,1)));
% figure; imagesc(makeimagestack(newvols(:,:,:,2)));

% inputs
if ~exist('figuredir','var') || isempty(figuredir)
  figuredir = [];
end
if ~exist('ref','var') || isempty(ref)
  ref = [1 1];
end
if ~exist('cutoff','var') || isempty(cutoff)
  cutoff = 1/90;
end
if ~exist('extratrans','var') || isempty(extratrans)
  extratrans = eye(4);
end
if ~exist('skipreslice','var') || isempty(skipreslice)
  skipreslice = 0;
end
if ~exist('realignparams','var') || isempty(realignparams)
  realignparams = struct('quality',1,'fwhm',3,'sep',2,'interp',7);
end
if ~exist('resliceparams','var') || isempty(resliceparams)
  resliceparams = struct('mask',1,'mean',0,'interp',7,'which',2);
end
if ~exist('binarymask','var') || isempty(binarymask)
  binarymask = [];
end
if ~exist('epiignoremcvol','var') || isempty(epiignoremcvol)
  epiignoremcvol = [];
end
if ~exist('dformat','var') || isempty(dformat)
  dformat = 'double';
end
if ~iscell(volsize)
  volsize = {volsize};
end
if ~iscell(epiignoremcvol)
  epiignoremcvol = {epiignoremcvol};
end
isbare = ~iscell(vols);
if isbare
  vols = {vols};
end
if length(volsize)==1
  volsize = repmat(volsize,[1 length(vols)]);
end
if length(epiignoremcvol)==1
  epiignoremcvol = repmat(epiignoremcvol,[1 length(vols)]);
end

% make figuredir if necessary
if ~isempty(figuredir)
  mkdirquiet(figuredir);
end

% figure out reference volumes
if isvector(ref)
  refvol = vols{ref(1)}(:,:,:,ref(2));
else
  refvol = ref;
end

% calc
xyzsize = sizefull(refvol,3);
numvols = length(vols);

% process each time-series
paramsB = {};
fprintf('processing %d time-series...',numvols);
parfor p=1:numvols
  
  % calc
  nt = size(vols{p},4);

  % make a temporary directory
  setrandstate;  % necessary because spm_realign sets the rand state to 0
  dir0 = maketempdir;
  
  % write data to temporary directory
  writespmfiles(cat(4,refvol,vols{p}),xyzsize,volsize{p}(1:3),[dir0 'image%06d']);  % NOTE: hard-coded with six digits.
  if ~isempty(binarymask)
    writespmfiles(binarymask,xyzsize,volsize{p}(1:3),[dir0 'binarymask']);
  end
  
  % figure out motion parameters [this writes new .hdr files]  [NOTE THAT SPM_REALIGN SETS THE RAND STATE TO 0]
  realignparams0 = realignparams;
  if ~isempty(binarymask)
    realignparams0.PW = subscript(matchfiles([dir0 'binarymask.img']),1,1);
  end
  spm_realign(matchfiles([dir0 'image*.img']),realignparams0);  % or 5?

  % read in motion parameters
  params = [];  % volumes x 6
  for q=1:nt+1
    if q==1
      data = spm_vol(sprintf([dir0 'image%06d.img'],q));
    else
      data(q) = spm_vol(sprintf([dir0 'image%06d.img'],q));
    end
    params(q,:) = spm_imatrix(data(q).mat);
  end
  
  % deal with epiignoremcvol
  if ~isempty(epiignoremcvol{p})
    params(1+epiignoremcvol{p},:) = NaN;
    for q=1:size(params,2)
      params(2:end,q) = nantoclosest(params(2:end,q));
    end
  end
  
  % filter motion parameters, ignoring the first
  paramsB{p} = params;
  paramsB{p}(2:nt+1,1:6) = tsfilter(params(2:nt+1,1:6)',constructbutterfilter1D(nt,cutoff * (nt*volsize{p}(4))),[1 0])';

  % display stuff
  if ~isempty(figuredir)
    figureprep([100 100 800 400]);
    
    subplot(1,2,1); hold on;
    plot(params(2:end,1)-params(1,1),'r-');
    plot(params(2:end,2)-params(1,2),'g-');
    plot(params(2:end,3)-params(1,3),'b-');
    plot(paramsB{p}(2:nt+1,1)-params(1,1),'r-','LineWidth',3);
    plot(paramsB{p}(2:nt+1,2)-params(1,2),'g-','LineWidth',3);
    plot(paramsB{p}(2:nt+1,3)-params(1,3),'b-','LineWidth',3);
    ax = axis; mx = max(abs([ax(3:4) 3]));
    axis([ax(1:2) -mx mx]);
    xlabel('volume number'); ylabel('translation (mm)');
    legend('x','y','z');
    title(sprintf('time-series %d of %d',p,numvols));
    
    subplot(1,2,2); hold on;
    plot(params(2:end,4)/pi*180,'r-');
    plot(params(2:end,5)/pi*180,'g-');
    plot(params(2:end,6)/pi*180,'b-');
    plot(paramsB{p}(2:nt+1,4)/pi*180,'r-','LineWidth',3);
    plot(paramsB{p}(2:nt+1,5)/pi*180,'g-','LineWidth',3);
    plot(paramsB{p}(2:nt+1,6)/pi*180,'b-','LineWidth',3);
    ax = axis; mx = max(abs([ax(3:4) 2]));
    axis([ax(1:2) -mx mx]);
    xlabel('volume number'); ylabel('rotation (deg)');
    legend('pitch','roll','yaw');

    figurewrite('motion%03d',p,[],figuredir);
  end
  
  % reslice?
  if ~skipreslice
  
    % write new motion parameters [writes .img and .hdr]
    for q=1:nt+1
      vol = loadbinary(sprintf([dir0 'image%06d.img'],q),'int16',xyzsize);
      data(q).mat = spm_matrix(paramsB{p}(q,:));
      if q==1  % only the first volume gets traveled away
        data(q).mat = data(q).mat * extratrans;
      end
      spm_write_vol(data(q),vol);
    end
    
    % let SPM do the interpolation [writes r*.img, r*.hdr]
    spm_reslice(matchfiles([dir0 'image*.img']),resliceparams);
  
    % read in the new volumes, ignoring the first one (since that's the reference)
    vols{p} = cast(loadbinary([dir0 'rimage*.img'],'int16',xyzsize,[],4),dformat);
    vols{p} = vols{p}(:,:,:,2:nt+1);
  
  end
  
  % clean up
  assert(rmdir(dir0,'s'));

end
fprintf('done.\n');

% prepare output
if isbare
  vols = vols{1};
end
