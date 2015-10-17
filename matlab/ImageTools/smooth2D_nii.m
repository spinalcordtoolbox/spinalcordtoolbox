function smooth2D_nii(fname,sigma,mask)
% smooth2D_nii(fname,sigma,mask)
% ('epi_half_ratio.nii.gz',2,mask)
nii=load_nii(fname);
if exist('mask','var'),
    mask=load_nii(mask);
    nii.img(~mask.img)=nan;
end
H = fspecial('Gaussian',[20 20],sigma);
figure
imagesc3D(nii.img),
nii.img=reshape2D_undo(nanconv(reshape2D(nii.img,1),H,'noedge'),1,nii.dims); % nanconv is in spinalcordtoolbox

figure
imagesc3D(nii.img,[0.5 1])

save_nii_v2(nii,[sct_tool_remove_extension(fname,1) '_smooth'])
