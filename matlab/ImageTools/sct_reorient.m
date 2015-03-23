function sct_reorient(nifti)
% sct_reorient(nifti)
% Example:
% sct_reorient('img.nii') --> output 'img_reorient.nii'
[basename,~, ext]=sct_tool_remove_extension(nifti,1);

nii=load_nii(nifti);
save_nii(nii, [basename '_reorient' ext]);
