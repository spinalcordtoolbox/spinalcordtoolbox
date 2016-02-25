function [vols,flt] = smoothvolumes(vols,volsize,fwhm)

% function [vols,flt] = smoothvolumes(vols,volsize,fwhm)
%
% <vols> are one or more 3D volumes concatenated along the fourth dimension
%   or a cell vector of things like that.
% <volsize> is a 3-element vector with the voxel size
% <fwhm> is a 3-element vector with the desired FWHM of a Gaussian filter
%
% smooth <vols> using a Gaussian filter (values below 0.01 are zeroed out).
% we perform the filtering using imfilter and the 'replicate' option.
% also, we handle NaNs intelligently by not allowing them to enter the 
% filtering operation.  however, voxels that are originally NaN are forced
% to be NaN in the output.
%
% we also return the filter in <flt>.  the values sum to 1.
%
% history:
% 2011/08/23 - now return <flt>
% 2011/03/08 - handle NaNs intelligently now. this changes old behavior.
%
% example:
% a = getsamplebrain;
% a(rand(size(a))>.9) = NaN;
% a2 = smoothvolumes(a,[2.5 2.5 2.5],[5 5 5]);
% figure; imagesc(makeimagestack(a));
% figure; imagesc(makeimagestack(a2));

% otherwise, do the normal case
sd = (fwhm ./ volsize)/(2*sqrt(2*log(2)));  % first convert to matrix units, then to standard deviations
flt = l1unitlength(constructsmoothingfilter(sd,0.01));
if iscell(vols)
  for zz=1:length(vols)
    for p=1:size(vols{zz},4)
      vols{zz}(:,:,:,p) = smoothvolumes_helper(vols{zz}(:,:,:,p),flt);
    end
  end
else
  for p=1:size(vols,4)
    vols(:,:,:,p) = smoothvolumes_helper(vols(:,:,:,p),flt);
  end
end

%%%%%

function f = smoothvolumes_helper(vol,flt)

% function f = smoothvolumes_helper(vol,flt)
%
% <vol> is a 3D volume, potentially with NaNs
% <flt> is a filter that sums to 1
%
% if there are no NaNs, just filter <vol> with <flt> using 'replicate','same','conv'.
% if there are NaNs, act like they don't exist.  however, make sure NaNs are returned
%   for all voxels that are originally NaN.
% return the filtered <vol>.

% which voxels are not NaNs?
good = ~isnan(vol);

% if no NaN are here, just do it the regular way
if all(good(:))
  f = imfilter(vol,flt,'replicate','same','conv');

% if there are NaNs, specially handle it
else

  % smooth, setting NaN to 0 so that they do not contribute to the sum
  volSM = imfilter(nanreplace(vol),flt,'replicate','same','conv');
  
  % what is the total integral of the smoothing filter for non-NaN values?
  % in the normal case, the sum will be 1.  if there are NaNs, the sum will be less than 1.
  volCT = imfilter(double(good),flt,'replicate','same','conv');
  
  % perform the weighted sum, making sure that NaN is produced when the smoothing filter covers only NaN values,
  % and making sure that voxels that were bad to start with end up as NaN.
  f = copymatrix(zerodiv(volSM,volCT,NaN,0),~good,NaN);

end
