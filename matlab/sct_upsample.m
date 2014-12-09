function sct_upsample(data_file)
% sct_upsample(data_file)
% sct_upsample qspace_crop_eddy_moco_norm
     disp(['Interpolate data by a factor of 2 ...'])
    
    data_file=sct_tool_remove_extension(data_file,1);
    [dmri_matrix,dims,scales] = read_avw(data_file); dmri_matrix(isnan(dmri_matrix))=0;
    dmri_matrix_interp = zeros(2*dims(1)-1,2*dims(2)-1,dims(3),dims(4));
    for iT = 1:dims(4)
        for iZ = 1:dims(3)
            dmri_matrix_interp(:,:,iZ,iT) =  interp2(squeeze(dmri_matrix(:,:,iZ,iT)),1,'spline');
        end
    end
    scales(1:2)=scales(1:2)/2;
    save_avw_v2(dmri_matrix_interp,[data_file '_interp'],'f',scales);
    
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