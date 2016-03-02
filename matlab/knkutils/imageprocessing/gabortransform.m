function f = gabortransform(x)

% function f = gabortransform(x)
%
% <x> is a square 2D image.  can have multiple images along the third dimension.
%
% return <f> as q x q x 8 x images with complex cell responses.
% for assumptions on the transformation, see the code.
% 
% example:
% im = imresize(getsampleimage,[128 128]);
% figure; imagesc(makeimagestack(gabortransform(im))); axis equal tight; colormap(hot);

% calc
res = size(x,1);
numim = size(x,3);

% constants
cpfov = res/4;
bandwidth = -1;%[-1.5 1.25];
numor = 8;
numph = 2;
thresh = 0.01;
scaling = 1;
mode = 0;
expt = 2;

% figure out spacing
[d,gbrs] = applymultiscalegaborfilters(randn(1,res*res),cpfov,bandwidth,{[1]},1,1,thresh,scaling,mode);
sz = size(gbrs{1},1);
if mod(sz,2)==0
  indices = {[1+(sz/2-1):res-sz/2]};
else
  indices = {[1+(sz-1)/2:res-(sz-1)/2]};
end
q = length(indices{1});

% do it
f = applymultiscalegaborfilters(squish(x,2)',cpfov,bandwidth,indices,numor,numph,thresh,scaling,mode);
f = permute(reshape(sum(reshape(f.^expt,numim,numph,[]),2),[numim numor q q]),[3 4 2 1]);


%%%%winnermap = generatewinnermap(squish(permute(reshape(f,numim,numor,q,q),[3 4 1 2]),2),cmaphue(numor));
