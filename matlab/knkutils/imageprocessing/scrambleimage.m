function f = scrambleimage(f,chunksize)

% function f = scrambleimage(f,chunksize)
%
% <f> is res x res x images
% <chunksize> is number of pixels along one side of a chunk.
%   res must be evenly divisible by <chunksize>.
%
% scramble each image independently.  we ensure that the image is actually
% scrambled (i.e. not identical to the original).
%
% example:
% figure; imagesc(scrambleimage(getsampleimage,100)); axis equal tight;

% calc
res = size(f,1);
numchunks = res/chunksize; assert(isint(numchunks));

% do it [COULD PROBABLY DO WITHOUT A FOR-LOOP, BUT OH WELL]
for p=1:size(f,3)
  f(:,:,p) = cell2mat(permutedim(mat2cell(f(:,:,p),repmat(chunksize,1,numchunks),repmat(chunksize,1,numchunks)),-1));
end
