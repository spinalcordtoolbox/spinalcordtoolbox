function sct_create_mask_bg_nii(fname,threshold)
A=load_untouch_nii(fname);

if ~exist('threshold','var'), threshold=0.3; end

mask = sct_create_mask_bg(A.img,threshold);

A.img=mask;
save_untouch_nii(A,[sct_tool_remove_extension(fname,1) '_bgmask.nii'])