function sct_register_SbS(src,dest)
% sct_register_SbS(src,dest)
% example: sct_register_SbS qspace.nii template.nii
[basename,path]=sct_tool_remove_extension(src,1);

dbstop if error
% move inputs to temp folder
tmp_folder=sct_tempdir;
sct_gunzip(src,tmp_folder,'src.nii');
sct_gunzip(dest,tmp_folder,'dest.nii');
cd(tmp_folder);

% register
sct_reslice src.nii dest.nii
sct_unix('sct_orientation -i dest.nii -s RPI -o dest.nii')
sct_unix('sct_register_multimodal -i src_reslice.nii -d dest.nii -p step=1,algo=slicereg2d_translation,gradStep=5,iter=100,metric=MeanSquares:step=2,algo=slicereg2d_affine,gradStep=30,iter=100,metric=MeanSquares:step=3,algo=slicereg2d_translation,metric=MeanSquares:step=4,algo=slicereg2d_bsplinesyn,metric=MeanSquares')
sct_unix('sct_concat_transfo -w step0/step00GenericAffine.mat,step0/step01Warp.nii.gz,warp_src_reslice2dest.nii.gz -d dest.nii -o warp_forward.nii')
sct_unix('sct_concat_transfo -w warp_dest2src_reslice.nii.gz,step0/step01InverseWarp.nii.gz,-step0/step00GenericAffine.mat -d src.nii -o warp_inverse.nii');

% bring back results
sct_unix(['mv src_reslice_reg.nii ' basename '_reg.nii']);
sct_unix(['mv warp_forward.nii ' path]);
sct_unix(['mv warp_inverse.nii ' path]);

cd ../
rmdir(tmp_folder,'s')
disp(['>> unix('' fslview ' basename '_reg.nii ' dest ''')'])
