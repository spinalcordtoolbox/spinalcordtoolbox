function sct_moco(fname, ref)
% sct_moco(fname, ref)

tmp=sct_tempdir;
sct_gunzip(fname,tmp,'data.nii')
cd(tmp)
sct_unix(['fslroi data.nii data_ref.nii ' num2str(ref-1) ' 1'])
sct_unix(['sct_image -i data_ref.nii.gz -setorient RPI'])
sct_unix(['sct_register_multimodal -i data.nii -d data_ref_RPI.nii.gz -p step=1,algo=slicereg2d_translation,metric=CC']);
cd ..
sct_unix(['cp ' tmp filesep 'data_reg.nii ' sct_tool_remove_extension(fname,1) '_moco.nii'])
unix(['rm -rf ' tmp])