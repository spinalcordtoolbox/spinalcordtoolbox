function f = makegratings2d(res,sfs,numor,numph)

% function f = makegratings2d(res,sfs,numor,numph)
%
% <res> is the number of pixels along one side
% <sfs> is a vector of spatial frequencies
% <numor> is positive number of orientations (starting at 0).
%   can be -Y where Y is a vector of orientations.
% <numph> is number of phases (starting at 0)
%
% return a series of 2D images where values are in [-1,1].
% the dimensions of the returned matrix are res x res x ph*or*sf
%
% example:
% figure; imagesc(makeimagestack(makegratings2d(32,[1 2],4,2)));

% calc
numsf = length(sfs);
if numor(1) <= 0
  ors = -numor;
  numor = length(ors);
else
  ors = linspacecircular(0,pi,numor);
end
phs = linspacecircular(0,2*pi,numph);

% do it
f = zeros(res,res,numph,numor,numsf);
xx = []; yy = [];
for p=1:numsf
  for q=1:numor
    for r=1:numph
      [f(:,:,r,q,p),xx,yy] = makegrating2d(res,sfs(p),ors(q),phs(r),xx,yy);
    end
  end
end
f = reshape(f,res,res,[]);
