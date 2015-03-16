function mask=sct_create_mask(fname)
% mask=sct_create_mask(fname)
% Call fslview

[basename, path]=sct_tool_remove_extension(fname,1);
if exist([basename '-mask.nii.gz'])
    mask=[basename '-mask.nii.gz'];
elseif ~isempty(dir([path 'mask*spinal*cord*']))
    mask=cell2mat(sct_tools_ls([path 'mask*spinal*cord*'],1));
else
    unix(['fslview ' fname])
    if exist([basename '-mask.nii.gz'])
        mask=[basename '-mask.nii.gz'];
    else
        [mask,mask_path] = uigetfile('*.nii;*.nii.gz','Select the mask you just created') ;
        mask = [mask_path,mask];
    end
end
