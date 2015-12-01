function [list, path]=sct_tools_ls(fname, keeppath, keepext, folders)
% [list, path]=sct_tools_ls(fname, keeppath?, keepext?, folders?)
% Example: sct_tools_ls('ep2d*')
% example 2: sct_tools_ls('*',[],[],1) --> folders only
% example 3: sct_tools_ls('*',[],[],2) --> files only

if nargin < 2, keeppath=0; end
if nargin < 3, keepext=1; end
if nargin < 4, folders=0; end
% [list, path]=sct_tools_ls('*T.txt);
list=dir(fname);
path=[fileparts(fname) filesep];
if strcmp(path,filesep)
    path=['.' filesep];
end

if folders==1
    list=list(cat(1,list.isdir));
elseif folders==2
    list=list(~cat(1,list.isdir));
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

if length(list)==1, list=list{1}; end