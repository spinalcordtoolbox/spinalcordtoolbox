function f = imtoch(f)

% function f = imtoch(f)
%
% <f> is ch x ch x images
%
% make <f> into images x ch*ch.
%
% example:
% isequal(size(imtoch(randn(10,10,2))),[2 100])

f = squish(f,2).';
