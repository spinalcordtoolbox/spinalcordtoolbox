function sct_downsampleinT(data,schemefile,factor)
%sct_downsampleinT(data,scheme,factor)
nii=load_nii(data);
save_nii_v2(nii.img(:,:,:,1:factor:end),[sct_tool_remove_extension(data,1) '_div' num2str(factor)],data);
scheme=scd_schemefile_read(schemefile);
scd_schemefile_write(scheme(1:factor:end,:),[sct_tool_remove_extension(schemefile,1) '_div' num2str(factor) '.scheme'])
