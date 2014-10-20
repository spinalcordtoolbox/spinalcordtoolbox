function [path, list]=sct_tools_ls(fname)
% [path, list]=sct_tools_ls('*T.txt);
list=dir(fname);
path=[fileparts(fname) filesep];
if strcmp(path,filesep)
    path=['.' filesep];
end
list=sort_nat({list.name});
