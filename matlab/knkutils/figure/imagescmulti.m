function imagescmulti(x)

% function imagescmulti(x)
%
% <x> is a 3D matrix with different images along the third dimension
%
% make a separate figure window for each image (using imagesc and axis equal tight).
%
% example:
% imagescmulti(randn(10,10,3));

for p=1:size(x,3)
  drawnow; figure; imagesc(x(:,:,p)); axis equal tight;
end
