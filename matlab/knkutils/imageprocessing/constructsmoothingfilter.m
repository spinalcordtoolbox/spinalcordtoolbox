function flt = constructsmoothingfilter(sd,thresh)

% function flt = constructsmoothingfilter(sd,thresh)
%
% <sd> is [A B C] with standard deviations of a 3D Gaussian filter along the
%   first, second, and third dimensions.  you can also specify [A B] in order
%   to obtain a 2D Gaussian filter.
% <thresh> is threshold below which to zero out values in the filter
%
% return a unit-length space-domain 3D (or 2D) Gaussian image filter.
% the filter has an odd size (and is therefore peaked at the central pixel).
%
% history:
% 2010/03/03 - revamp entirely (changed input format) --- we now support 3D.
%
% example:
% a = getsampleimage;
% flt = constructsmoothingfilter([10 10],0.01);
% figure; imagesc(a); axis equal tight;
% figure; imagesc(imfilter(a,flt,'replicate','same','conv')); axis equal tight;

% make the filter
temp = 10;
while 1
  if length(sd)==2
    flt = makegaussian3d([temp temp 1],repmat((temp/2 - 1)/(temp-1),[1 3]),[sd 1]/(temp-1));
    flt(flt < thresh) = 0;
    if all(flatten(flt(1,:,:))==0) && all(flatten(flt(end,:,:))==0) && ...
       all(flatten(flt(:,1,:))==0) && all(flatten(flt(:,end,:))==0)
      break;
    else
      temp = temp*2;
    end
  else
    flt = makegaussian3d([temp temp temp],repmat((temp/2 - 1)/(temp-1),[1 3]),sd/(temp-1));
    flt(flt < thresh) = 0;
    if all(flatten(flt(1,:,:))==0) && all(flatten(flt(end,:,:))==0) && ...
       all(flatten(flt(:,1,:))==0) && all(flatten(flt(:,end,:))==0) && ...
       all(flatten(flt(:,:,1))==0) && all(flatten(flt(:,:,end))==0)
      break;
    else
      temp = temp*2;
    end
  end
end

% TODO: use cropvalidvolume.m?

% crop the filter
goodrows = find(~all(all(flt==0,2),3));
goodcols = find(~all(all(flt==0,1),3));
gooddeps = find(~all(all(flt==0,1),2));
flt = flt(goodrows,goodcols,gooddeps);

% check that filter has odd number of pixels
assert(mod(size(flt,1),2)==1);
assert(mod(size(flt,2),2)==1);
assert(mod(size(flt,3),2)==1);

% make unit-length
flt = unitlength(flt);






% % input
% if ~exist('wantcheck','var') || isempty(wantcheck)
%   wantcheck = 0;
% end
% 

% % show figures (USE VIEWIMAGE, see fouriertospace!)
% if wantcheck
%   temp = fftshift(abs(fft2(placematrix(zeros(size(flt,1),size(flt,1)),flt,[1 1]))));
%   figure; imagesc(flt); axis equal tight; title('filter');
%   figure; plot(flt(round(end/2),:),'r.-'); title('profile through filter');
%   figure; imagesc(temp); axis equal tight; title('amplitude spectrum of filter');
% end
