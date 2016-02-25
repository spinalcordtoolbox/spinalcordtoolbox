function [f,images] = applyfiltermultidim(images,numfeatures,x,alreadyfft2)

% function [f,images] = applyfiltermultidim(images,numfeatures,x,alreadyfft2)
%
% <images> is images x <numfeatures>*res*res.
%   alternatively, can be the result of:
%     images = fft2(permute(reshape(images,[],res,res),[2 3 1]));  % res x res x images*numfeatures
%   in this case, pass in <alreadyfft2> as 1.
% <numfeatures> is the number of features associated with each position
% <x> is a 3D matrix with the filter (<numfeatures> x N x N).
%   can be passed in as a vector.
% <alreadyfft2> (optional) is whether <images> has already been subject to fft2.
%   default: 0.
%
% convolve filter <x> across the images.  we do not consider filter outputs that
% extend beyond the bounds of the images (i.e. the 'valid' option
% in filter2.m).  return filter outputs in <f>, a matrix of dimensions
% images x S*S where S is res-N+1.  also, return <images> after being subject
% to fft2.m (see above).
%
% example:
% f = applyfiltermultidim(randn(1,4*10*10),4,randn(4,2,2));

% input
if ~exist('alreadyfft2','var') || isempty(alreadyfft2)
  alreadyfft2 = 0;
end

% calc and prep
if alreadyfft2
  res = size(images,1);
  numim = size(images,3)/numfeatures; assert(isint(numim));
else
  res = sqrt(size(images,2)/numfeatures); assert(isint(res));
  numim = size(images,1);
end
n = sqrt(numel(x)/numfeatures); assert(isint(n));
s = res-n+1;
x = reshape(x,[numfeatures n n]);

% do it
if ~alreadyfft2
  images = fft2(permute(reshape(images,numim*numfeatures,res,res),[2 3 1]));  % res x res x images*numfeatures
end
f = 0;
for pp=1:numfeatures
  f = f + real(ifft2(bsxfun(@times,images(1:res,1:res,(pp-1)*numim+(1:numim)),fft2(placematrix(zeros(res,res),rot90(squish(x(pp,:,:),2),2),[1 1])))));  % res x res x images
end
f = reshape(permute(f(n:end,n:end,:),[3 1 2]),numim,[]);




% OLD SLOW WAY:
% % calc and prep
% res = sqrt(size(images,2)/numfeatures); assert(isint(res));
% n = sqrt(numel(x)/numfeatures); assert(isint(n));
% s = res-n+1;
% x = reshape(x,[numfeatures n n]);
% 
% % do it
% f = zeros(size(images,1),s,s,numfeatures);
% for pp=1:size(images,1)
%   im = reshape(images(pp,:),numfeatures,res,res);
%   for qq=1:numfeatures
%     f(pp,:,:,qq) = filter2(squish(x(qq,:,:),2),squish(im(qq,:,:),2),'valid');
%   end
% end
% f = reshape(sum(f,4),size(images,1),[]);




% WELL, NO NEED TO CONSTRUCT IT!
% % construct f
% f = zeros(numfeatures*res*res,numfilters);
% ix0 = false(numfeatures,res,res);  % index that will indicate which features are actually involved in a filter
% for c=1:g  % columns
%   for r=1:g  % rows
%     ix = ix0;
%     ix(:,r:r+n-1,c:c+n-1) = true;
%     f(ix(:),(c-1)*g+r) = x;
%   end
% end

% A FAILED ATTEMPT TO SPEED UP
% 
% 
% bb = permute(reshape(x,[numfeatures n n]),[2 3 1]);
% 
% f2 = zeros(res,n,numfeatures,g);
% for p=1:g
%   f2(p:p+n-1,1:n,1:numfeatures,p) = bb;
% end
% 
% f = zeros(res,res,numfeatures,g*g);
% for p=1:g
%   f(1:res,p:p+n-1,1:numfeatures,(p-1)*g+(1:g)) = f2;
% end
% 
% f = squish(permute(f,[3 1 2 4]),3);
