function [basename, varargout]=sct_tool_remove_extension(fname,keeppath)
% [basename(,path, ext)]=sct_tool_remove_extension(fname,keeppath?)
% e.g. : 'epi2d'=sct_tool_remove_extension('data/epi2d.nii.gz',0)
% e.g. : 'data/epi2d'=sct_tool_remove_extension('data/epi2d.nii.gz',1)

    [path, name, ext1]=fileparts(fname);
    [~, basename, ext2]=fileparts(name);
    
    if isempty(path)
        path=pwd;
    end
    
    if keeppath
        basename=[path filesep basename];
    end
    
    varargout{1}=[path filesep];
    varargout{2}=[ext2 ext1];
end