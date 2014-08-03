function basename=sct_tool_remove_extension(fname,keeppath)
% basename=sct_tool_remove_extension(fname,keeppath?)
% e.g. : 'epi2d'=sct_tool_remove_extension('data/epi2d.nii.gz',0)
    [path, name]=fileparts(fname);
    [~, basename]=fileparts(name);
    if keeppath
        basename=[path filesep basename];
    end
end