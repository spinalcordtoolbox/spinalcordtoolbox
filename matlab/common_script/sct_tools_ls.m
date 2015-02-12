function [list, path]=sct_tools_ls(fname, keeppath)
% [list, path]=sct_tools_ls(fname, keeppath?)
if nargin < 2, keeppath=0; end
% [list, path]=sct_tools_ls('*T.txt);
list=dir(fname);
path=[fileparts(fname) filesep];
if strcmp(path,filesep)
    path=['.' filesep];
end
% sort by name
list=sort_nat({list.name});
% remove files starting with .
list(cellfun(@(x) strcmp(x(1),'.'), list))=[];
if keeppath
    for iL=1:length(list)
        list{iL}=[path list{iL}];
    end
end