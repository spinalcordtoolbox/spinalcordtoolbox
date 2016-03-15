function f = chtoim(f)

% function f = chtoim(f)
%
% <f> is images x ch*ch
%
% make <f> into ch x ch x images.
%
% example:
% isequal(size(chtoim(randn(2,100))),[10 10 2])

ch = sqrt(size(f,2));
f = reshape(f.',ch,ch,[]);
