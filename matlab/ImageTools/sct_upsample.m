function sct_upsample(data_file)
% sct_upsample(data_file)
% sct_upsample qspace_crop_eddy_moco_norm
     disp(['Interpolate data by a factor of 2 ...'])
    nii = load_nii(data_file); 
    data_file=sct_tool_remove_extension(data_file,1);
    dmri_matrix=nii.img; dmri_matrix(isnan(dmri_matrix))=0;
    dmri_matrix_interp = zeros(2*nii.dims(1)-1,2*nii.dims(2)-1,nii.dims(3),nii.dims(4));
    for iT = 1:nii.dims(4)
        for iZ = 1:nii.dims(3)
            dmri_matrix_interp(:,:,iZ,iT) =  interp2(squeeze(dmri_matrix(:,:,iZ,iT)),1,'spline');
        end
    end
    nii.hdr.dime.pixdim(2:3)=nii.hdr.dime.pixdim(2:3)/2;
    nii.img=dmri_matrix_interp;
    save_nii_v2(nii,[data_file '_interp']);
    
    % change the default data file name
    data_file = [data_file '_interp'];
    disp(['... File created: ',data_file])
    
    
    
%     function sct_upsample(data_file)
% % sct_upsample(data_file)
% % sct_upsample qspace_crop_eddy_moco_norm
%      disp(['Interpolate data by a factor of 2 ...'])
%     unix(['sct_resample -i ' data_file ' -f 2x2x1']);
%      
%     data_file=sct_tool_remove_extension(data_file,1);
%     
%     unix(['mv -f ' data_file 'r.nii* ' data_file '_interp.nii']);
%     
%     % change the default data file name
%     data_file = [data_file '_interp'];
%     disp(['... File created: ',data_file])