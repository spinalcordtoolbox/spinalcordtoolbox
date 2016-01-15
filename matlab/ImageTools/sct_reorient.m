function nifti_reorient=sct_reorient(nifti)
% sct_reorient(nifti)
% Example:
% sct_reorient('img.nii') --> output 'img_RPI.nii'
[basename,~, ext]=sct_tool_remove_extension(nifti,1);
cmd=['sct_image -i ' nifti ' -setorient RPI'];
unix(cmd)

nifti_reorient=[basename '_RPI' ext];