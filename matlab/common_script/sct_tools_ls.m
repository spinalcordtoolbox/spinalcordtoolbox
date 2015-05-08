function [list, path]=sct_tools_ls(fname, keeppath, keepext)
% [list, path]=sct_tools_ls(fname, keeppath?, keepext?)
if nargin < 2, keeppath=0; end
if nargin < 3, keepext=1; end
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

if ~keepext
    list=cellfun(@(x) sct_tool_remove_extension(x,1),list,'UniformOutput',false);
end