function [newvols,voloffset,validvol] = undistortvolumes(vols,volsize,pixelshifts,phasedir,mcparams,extratrans,targetres)

% function [newvols,voloffset,validvol] = undistortvolumes(vols,volsize,pixelshifts,phasedir,mcparams,extratrans,targetres)
%
% <vols> is one or more 3D volumes concatenated along the fourth dimension.
%   the number of volumes is N.  it is okay if <vols> is int16.
% <volsize> is a 3-element vector with the size of each voxel in mm
% <pixelshifts> is a 3D volume with the number of pixel shifts that the magnetic 
%   field caused.  the dimensions of <pixelshifts> should match the volumes in 
%   <vols>.  special case is [] which means to omit undistortion.
%   can also be a 4D volume which allows a distinct shift for each time point.
% <phasedir> is an integer indicating the phase-encode direction.
%   valid values are 1, -1, 2, or -2.  a value of 1 means the phase-encode 
%   direction is oriented along the first matrix dimension (e.g. 1->64); 
%   -1 means the reverse (e.g. 64->1).  a value of 2 means the phase-encode
%   direction is oriented along the second matrix dimension (e.g. 1->64);
%   -2 means the reverse (e.g. 64->1).  does not matter if <pixelshifts> is [].
% <mcparams> is an N x 12 matrix with SPM's motion parameter estimate for
%   each volume.  special case is [] which means to omit motion correction.
% <extratrans> (optional) is a 4x4 transformation matrix that maps points in the
%   matrix space of the 3D volumes to a new location.  if supplied, then volumes 
%   will be resampled at the new location.  for example, if <extratrans>
%   is [1 0 0 1; 0 1 0 0; 0 0 1 0; 0 0 0 1], then this will cause volumes to be 
%   resampled at a location corresponding to a one-voxel shift along the first
%   dimension.  <extratrans> can also be {X} where X is 4 x vertices, indicating
%   the exact locations (relative to the matrix space of the 3D volume) at which
%   to sample the data.  in this case, <targetres> must be [].  default: eye(4).
% <targetres> (optional) is
%   (1) [X Y Z], a 3-element vector with the number of voxels desired for the final 
%       output.  if supplied, then volumes will be interpolated only at the points 
%       necessary to achieve the desired <targetres>.  we assume that the field-
%       of-view is to be preserved.
%   (2) {[A B C] [D E F] G}, where [A B C] is a 3-element vector with the number of voxels
%       desired for the final output and [D E F] is the voxel size in mm.  in this case, 
%       we do not assume the field-of-view is preserved; rather, we just assume the 
%       desired grid is ndgrid(1:A,1:B,1:C).  G is 0/1 indicating whether to tightly crop 
%       the output volumes to the smallest 3D box that contains all the non-NaN values 
%       in the very first corrected volume.
%   default is [] which means to do nothing special.
%
% resample <vols> to undo the effects of distortion and motion and also
% accommodate <extratrans> and <targetres>.  we assume that motion parameter 
% estimates pertain to the volumes after undistortion is applied to each volume.  
% note that returned volumes may have NaN in them (when we lack data at locations to
% be sampled)!  the format of the returned volumes is int16.  because of limitations
% in the int16 format, what should have been NaNs gets returned as 0s.  if you need
% to be careful, examine the contents of the <validvol> output variable, which is
% which has the dimensions of one volume.  <validvol> is a logical matrix indicating 
% which voxels would have had no NaNs in their time-series if we didn't do the 
% int16 conversion.
%
% the <voloffset> output is a 3-element vector of non-negative integers indicating
% the offset relative to the original volume.  when <targetres> is [] or [X Y Z] or
% {[A B C] [D E F] 0}, <voloffset> is always returned as [0 0 0].  it is only when <targetres> 
% is {[A B C] [D E F] 1} that <voloffset> may have an interesting value.  for example, 
% [100 0 40] means that the first voxel is actually (101,1,41) of the original volume.
%
% we use cubic interpolation as implemented in ba_interp3 (instead of interp3).
% in particular, note that ba_interp3 repeats the border values, which is different
% from interp3 (i think).  also, note that we explicitly ensure that points outside 
% the original volume are given NaN values.
%
% we use parfor to speed up execution.
%
% history:
% 2014/09/16 - allow <extratrans> to be the {X} case
% 2011/03/19 - final polishing of recent changes.  these included: switch to ba_interp3; use ba_interp3 for forward distortion; new output validvol; explicit on data format issues; more flexible input; int16 for the output
% 2011/03/16 - allow <pixelshifts> to be 4D
% 2011/03/15 - report some messages to stdout; use ba_interp3_wrapper; switch to ba_interp3 for forward distortion calculation
% 2011/03/14 - return <newvols> as int16 format.  return new output <validvol>.
% 2011/03/10 - implement new case of <targetres> and the output <voloffset>
% 2011/03/06 - we now use parfor to speed up execution.
% 2011/03/03 - use ba_interp3 instead of interp3
%
% example:
% vol = getsamplebrain(1);
% volB = undistortvolumes(vol,[2.5 2.5 2.5],rand(64,64,19)*4,2,[]);
% figure; imagesc(makeimagestack(vol));
% figure; imagesc(makeimagestack(volB));

% report
fprintf('undistortvolumes called for a <vols> of size %s...',mat2str(sizefull(vols,4))); stime = clock;

% input
if ~exist('extratrans','var') || isempty(extratrans)
  extratrans = eye(4);
end
if ~exist('targetres','var') || isempty(targetres)
  targetres = sizefull(vols,3);
end

% calc
voldim = sizefull(vols,3);
wantspecialcrop = iscell(targetres) && targetres{3}==1;

% deal with special vertex/flat case
if iscell(extratrans)
  targetres = [size(extratrans{1},2) 1];
  dimdata = 1;
else
  dimdata = 3;
end

% construct coordinates (which are always in matrix space)
if iscell(extratrans)
  coords = extratrans{1};
  extratrans = 1;  % necessary hack because later we will do inv(extratrans)
else
  if iscell(targetres)
    targetres = targetres{1};  % now, targetres is always a 3-element vector.  note that we ignore targetres{2} currently!
    [xx,yy,zz] = ndgrid(1:targetres(1), ...
                        1:targetres(2), ...
                        1:targetres(3));
  else
    [xx,yy,zz] = ndgrid(resamplingindices(1,size(vols,1),targetres(1)), ...
                        resamplingindices(1,size(vols,2),targetres(2)), ...
                        resamplingindices(1,size(vols,3),targetres(3)));
  end
  coords = [flatten(xx); flatten(yy); flatten(zz); ones(1,numel(xx))];
end

% here's some idea on the tricky ordering issues:
% - in the reverse direction, we would first undistort each volume,
%   take each volume and map it to the reference's volume space,
%   and then take each volume and put it in the final extratrans space.
% - so, our strategy is to do this in the forward direction.  suppose we
%   start in the final extratrans space.  then what we need to do
%   is to undo the extratrans transformation.  then we need to
%   perform the transformation takes each volume to its motion-
%   corrupted location.  then we finally need to perform the distortion.

% transform coordinates for the full case
if ~isempty(mcparams)

  % construct SPM's transformation matrix
  spmt = createspmmatrix(voldim,volsize);
  
  % special initial handling for the special crop case in order to figure out the crop
  if wantspecialcrop
  
    % THIS IS A BIT HACKY, AS WE ARE COPYING THE INNER FOR-LOOP BELOW
    p = 1;
    wh = choose(size(pixelshifts,4)>1,p,1);
    coordsB = computeforwarddistortion(computeforwardmotion(inv(extratrans)*coords,spmt,mcparams(p,:)),pixelshifts(:,:,:,wh),phasedir);
      
    % aha, now, do the cropping
    [temp,voloffset] = cropvalidvolume(reshape(ba_interp3_wrapper(vols(:,:,:,p),coordsB),targetres),@(x) ~isnan(x));
    
    % now, re-generate the voxel locations that we need
    [xx,yy,zz] = ndgrid(voloffset(1)+(1:size(temp,1)), ...
                        voloffset(2)+(1:size(temp,2)), ...
                        voloffset(3)+(1:size(temp,3)));
    coords = [flatten(xx); flatten(yy); flatten(zz); ones(1,numel(xx))];
    
    % finally, simulate the new targetres
    targetres = sizefull(temp,3);

  % otherwise, just set the voloffset output here
  else
    voloffset = [0 0 0];
  end
  
  % do it
  newvols = zeros([size(vols,4) targetres],'int16');
  validvol = true(targetres);
  parfor p=1:size(mcparams,1)
%     stimeB = clock;

    % undo the extratrans, pull from motion-corrupted location, but actually pull from pixelshifted location
    wh = choose(size(pixelshifts,4)>1,p,1);
    coordsB = computeforwarddistortion(computeforwardmotion(inv(extratrans)*coords,spmt,mcparams(p,:)),pixelshifts(:,:,:,wh),phasedir);

    % resample the volume
    temp = reshape(ba_interp3_wrapper(vols(:,:,:,p),coordsB),targetres);
    validvol = validvol & ~isnan(temp);  % it must be the case that the value in the new volume is not nan
    newvols(p,:,:,:) = temp;

%     % report
%     fprintf('(volume %d took %.1f min) ',p,etime(clock,stimeB)/60);

  end

% transform coordinates for the case of no motion
else

  % special initial handling for the special crop case in order to figure out the crop
  if wantspecialcrop

    % THIS IS A BIT HACKY, AS WE ARE COPYING THE CODE BELOW [MORE OR LESS]
    p = 1;
    coordsB = computeforwarddistortion(inv(extratrans)*coords,pixelshifts(:,:,:,1),phasedir);
    
    % aha, now, do the cropping
    [temp,voloffset] = cropvalidvolume(reshape(ba_interp3_wrapper(vols(:,:,:,p),coordsB),targetres),@(x) ~isnan(x));

    % now, re-generate the voxel locations that we need
    [xx,yy,zz] = ndgrid(voloffset(1)+(1:size(temp,1)), ...
                        voloffset(2)+(1:size(temp,2)), ...
                        voloffset(3)+(1:size(temp,3)));
    coords = [flatten(xx); flatten(yy); flatten(zz); ones(1,numel(xx))];
    
    % finally, simulate the new targetres
    targetres = sizefull(temp,3);

  % otherwise, just set the voloffset output here
  else
    voloffset = [0 0 0];
  end

  % if we have a single pixelshifts, we can compute this here:
  % compute coordinates by undoing the extratrans and then pulling from pixelshifted location
  if size(pixelshifts,4)==1
    coordsB0 = computeforwarddistortion(inv(extratrans)*coords,pixelshifts,phasedir);
  else
    coordsB0 = [];  % just a placeholder
  end

  % resample the volumes
  newvols = zeros([size(vols,4) targetres],'int16');
  validvol = true(targetres);
  parfor p=1:size(vols,4)
%     stimeB = clock;
    if size(pixelshifts,4)==1
      coordsB = coordsB0;
    else
      coordsB = computeforwarddistortion(inv(extratrans)*coords,pixelshifts(:,:,:,p),phasedir);
    end
    temp = reshape(ba_interp3_wrapper(vols(:,:,:,p),coordsB),targetres);
    validvol = validvol & ~isnan(temp);  % it must be the case that the value in the new volume is not nan
    newvols(p,:,:,:) = temp;
%     fprintf('(volume %d took %.1f min) ',p,etime(clock,stimeB)/60);
  end
  
  % NOTE: this case could be even faster by taking advantage of ba_interp3's multiple channel reslicing thing.

end

% prepare output
if dimdata==3
  newvols = permute(newvols,[2 3 4 1]);
else
  newvols = permute(newvols,[2 1]);
end

% report
fprintf('done (elapsed time %.1f minutes).\n',etime(clock,stime)/60);

%%%%%

function coords = computeforwardmotion(coords,spmt,mcparams)

% function coords = computeforwardmotion(coords,spmt,mcparams)
%
% <coords> is 4 x N with coordinates in matrix space
% <spmt> is the 4 x 4 SPM transformation matrix that goes from matrix space
%   to SPM's internal space
% <mcparams> is a 12-element vector with SPM's motion parameter estimate.
%   spm_matrix(<mcparams>) should tell us how to go from the matrix space of
%   the current volume to SPM's internal space.
%
% return new coordinates after moving according to <mcparams>.

% the order is: go from matrix space to SPM's internal space;
%               go from SPM's internal space to the motion-corrupted EPI space;
coords = (inv(spm_matrix(mcparams))*spmt)*coords;

%%%%%

function coords = computeforwarddistortion(coords,pixelshifts,phasedir)

% function coords = computeforwarddistortion(coords,pixelshifts,phasedir)
%
% <coords> is 4 x N with coordinates in matrix space
% <pixelshifts> is like in undistortvolumes.m, except that we have to be
%   a single 3D volume
% <phasedir> is like in undistortvolumes.m
%
% return new coordinates after performing the distortion described by
% <pixelshifts>.  if any of the original <coords> is located 
% outside the FOV of <pixelshifts>, then the first or second component
% of the new coordinates will be NaN (indicating that we should not lookup
% a value for those coordinates).

% if no pixelshifts, then do nothing
if isempty(pixelshifts)
  return;
end

% determine the value of pixelshifts to use via interpolation
vi = ba_interp3_wrapper(pixelshifts,coords);

% shift the appropriate coordinate component
switch phasedir
case 1
  coords(1,:) = coords(1,:) + vi;
case -1
  coords(1,:) = coords(1,:) - vi;
case 2
  coords(2,:) = coords(2,:) + vi;
case -2
  coords(2,:) = coords(2,:) - vi;
end











%%%%%%%%%%%%%%%%%%%%%%%%%% JUNK:

%     newvols(:,:,:,p) = reshape(copymatrix(interp3(vols(:,:,:,p),coordsB(2,:),coordsB(1,:),coordsB(3,:),'cubic',NaN), ...
%                                        bad,NaN),targetres);
%     newvols(:,:,:,p) = reshape(copymatrix(interp3(vols(:,:,:,p),coordsB(2,:),coordsB(1,:),coordsB(3,:),'cubic',NaN), ...
%                                        bad,NaN),targetres);
