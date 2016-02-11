function sct_create_mask_bg_nii(fname,threshold)
% sct_create_mask_bg_nii(fname,threshold)
% input: a NIFTI image filename. 
% Output : create a binary mask NIFTI file with value 1 in the region of
% interest. 0 in the baskground. Threshold is a percentage ([0 1]) 
A=load_untouch_nii(fname);

if ~exist('threshold','var'), threshold=0.3; end

mask = sct_create_mask_bg(A.img,threshold);

A.img=mask;
save_untouch_nii(A,[sct_tool_remove_extension(fname,1) '_bgmask.nii'])