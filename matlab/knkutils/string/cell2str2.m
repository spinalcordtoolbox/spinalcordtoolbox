function f = cell2str2(m)

% function f = cell2str2(m)
%
% <m> is a cell vector of strings like {'apple' 'banana'}
%
% return a string that names each string, like
%   ,'apple','banana'
% this function is useful for saving variables to .mat files.
%
% example:
% a = {'apple' 'banana'}
% cell2str2(a)

f = catcell(2,cellfun(@(x) [',''' x ''''],m,'UniformOutput',0));
