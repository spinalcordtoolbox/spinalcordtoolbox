function f = coregistervolumes(m,len,wantmi)

% function f = coregistervolumes(m,len,wantmi)
%
% <m> is a set of 3D volumes concatenated along the fourth dimension.
%   there should be at least two 3D volumes.  the volumes should be
%   suitable for interpretation as int16.
% <len> is a 3-element vector with the matrix lengths in millimeters
% <wantmi> (optional) is whether to want to use mutual information
%   as a metric.  if 1, then we will use spm_coreg to figure out
%   transformation parameters; if 0, then we will use spm_realign.m.
%   default: 0.
%
% coregister all volumes to the first volume.  return the resliced volumes
% in double format.  we perform coregistration by using routines from SPM.
% note that voxels that move outside of the field-of-view will have 0s
% in all volumes.
%
% note that we use some defaults for the SPM function calls (see code).
% also we assume SPM writes out 'int16' files.
%
% note that the SPM knob values are extremely important to be aware of!!
% weird things can happen if the input data and the knob values are
% mismatched!
%
% we have tested only SPM5.  no guarantees on whether this works for other versions of SPM.
%
% example:
% a = getsamplebrain * 1000;
% b = circshift(a + randn(size(a)) * 100,[4 2]);
% b2 = coregistervolumes(cat(4,a,b),[3 3 3],0);
% b3 = coregistervolumes(cat(4,a,b),[3 3 3],1);
% figure; imagesc(makeimagestack(a));
% figure; imagesc(makeimagestack(b));
% figure; imagesc(makeimagestack(b2(:,:,:,2)));
% figure; imagesc(makeimagestack(b3(:,:,:,2)));
%
% see also motioncorrectvolumes.m.

% input
if ~exist('wantmi','var') || isempty(wantmi)
  wantmi = 0;
end

% calc
msize = sizefull(m,3);

% make a temporary directory
setrandstate;  % necessary because spm_realign sets the rand state to 0
dir0 = maketempdir;

% write data to temporary directory
writespmfiles(m,msize,len,[dir0 'images%06d']);
files = matchfiles([dir0 'image*.img']);

% if wantmi, figure out transformations using spm_coreg
if wantmi

    % figure out params
  params = [0 0 0 0 0 0];  % the first entry
  for p=2:length(files)
    fprintf('*** calling spm_coreg for %d of %d.\n ***',p,length(files));
    params(p,:) = spm_coreg(files{p},files{1},struct('sep',[4 2]));  % let's make sep explicit because things really depend on it!
  end
    % now use params and hack in the transformations
  for p=1:length(files)
    writespmfiles(loadbinary(files{p},'int16',msize),msize,len,sprintf([dir0 'images%06d'],p),params(p,:));
  end

% otherwise, figure out transformations using spm_realign [this writes new .hdr files] [NOTE THAT SPM_REALIGN SETS THE RAND STATE TO 0]
else
  spm_realign(files,struct('quality',1,'fwhm',3,'sep',2,'interp',7));
end

% let SPM do the interpolation [writes r*.img, r*.hdr]
spm_reslice(matchfiles([dir0 'image*.img']),struct('mask',1,'mean',0,'interp',7,'which',2));

% read in the new volumes
f = double(loadbinary([dir0 'rimage*.img'],'int16',msize,[],4));



% OLD
%    V = spm_vol(files{p});
%    V.mat = spm_matrix(params(p,:)) * createspmmatrix(msize,len);
%    spm_write_vol(V,loadbinary(files{p},'int16',msize));
