function saveemptymat(file0)

% function saveemptymat(file0)
%
% <file0> is a file location ending in .mat
%
% save an empty .mat file to <file0>.
%
% example:
% saveemptymat('test.mat');
% a1 = load('test.mat')

ss = struct; 
save(file0,'-struct','ss');
