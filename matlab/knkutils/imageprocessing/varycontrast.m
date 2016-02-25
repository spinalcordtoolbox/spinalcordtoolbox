function f = varycontrast(f,cons,mode)

% function f = varycontrast(f,cons,mode)
%
% <f> is res x res x images
% <cons> is a vector of contrast values in [0,100]
% <mode> (optional) is
%   0 means concatenate along the third dimension
%   1 means make a cell vector
%   default: 0.
%
% reproduce the ensemble of stimuli at these contrast values.
% we scale with respect to the center value of 0.5.
%
% example:
% figure; imagesc(makeimagestack(varycontrast(getsampleimage,[10 20 50 100]))); axis equal tight;

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% do it
res = size(f,1);
temp = bsxfun(@times,reshape(cons/100,1,1,1,[]),f-.5) + .5;
if mode == 0
  f = reshape(temp,res,res,[]);
else
  f = splitmatrix(temp,4);
end
