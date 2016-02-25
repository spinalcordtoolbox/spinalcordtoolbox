function f = makecolorimagestack(varargin)

% function f = makecolorimagestack(varargin)
%
% input arguments are the same as for makeimagestack.m, except that
% <m> consists of one or more color images concatenated along the fourth dimension.
%
% we pass each color channel of <m> to makeimagestack.m and concatenate the results
% along the third dimension.
%
% example:
% figure; imagesc(makecolorimagestack(rand(10,10,3,4)));

f = [];
for p=1:size(varargin{1},3)
  f = cat(3,f,makeimagestack(permute(varargin{1}(:,:,p,:),[1 2 4 3]),varargin{2:end}));
end
