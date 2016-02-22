function sct_reslice(src,dest,pointwise_reg)
% sct_reslice(src,dest)
% EXAMPLE: sct_reslice('highQ_mean.nii','template_roi.nii')
unix(['isct_antsRegistration --dimensionality 3 --transform syn[0.5,3,0] --metric MI[' dest ',' src ',1,32] --convergence 0 --shrink-factors 1 --smoothing-sigmas 0mm --restrict-deformation 1x1x0 --output [step0,' sct_tool_remove_extension(src,1) '_reslice.nii] --interpolation BSpline[3] -r [' dest ',' src ',0]'])
src_reslice=[sct_tool_remove_extension(src,1) '_reslice.nii'];

mkdir step0
unix('mv step00GenericAffine.mat step0/')
unix('mv step01InverseWarp.nii.gz step0/')
unix('mv step01Warp.nii.gz step0/')

%% REGISTER USING CENTERLINES
if exist('pointwise_reg','var') && pointwise_reg
    tmp_folder=sct_tempdir;
    copyfile(src_reslice,[tmp_folder '/src_reslice.nii'])
    sct_gunzip(dest,tmp_folder, 'dest.nii')
    cd(tmp_folder)
    sct_unix('sct_image -i dest.nii -setorient RPI')

    %%
    sct_get_centerline_manual('src_reslice.nii')
    sct_get_centerline_manual('dest.nii')
    sct_unix('sct_image -i dest_centerline.nii -setorient RPI')
    
    
    sct_unix(['sct_register_multimodal -i src_reslice.nii -d dest_RPI.nii -iseg src_reslice_centerline.nii -dseg dest_centerline_RPI.nii -p step=1,type=seg,algo=slicereg2d_pointwise'])
    
    sct_unix('sct_concat_transfo -d dest_RPI.nii -w ../step0/step01Warp.nii.gz,../step0/step00GenericAffine.mat,warp_src_reslice2dest_RPI.nii.gz -o ../step0/step01Warp.nii.gz');
    sct_unix('sct_concat_transfo -d src_reslice.nii -w warp_dest_RPI2src_reslice.nii.gz,../step0/step01InverseWarp.nii.gz,-../step0/step00GenericAffine.mat -o ../step0/step01InverseWarp.nii.gz')
    
    cd ..
    unix(['rm -rf ' tmp_folder])
    unix('rm step0/step00GenericAffine.mat')
end
