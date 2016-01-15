function sct_create_mask_bg(fname,threshold)
A=load_untouch_nii(fname);

if ~exist('threshold','var'), threshold=0.3; end

mask=false(size(A.img));
for iz=1:size(A.img,3)
    mask(:,:,iz)=A.img(:,:,iz)>threshold*max(max(A.img(:,:,iz)));
end

A.img=mask;
save_untouch_nii(A,[sct_tool_remove_extension(fname,1) '_bgmask.nii'])