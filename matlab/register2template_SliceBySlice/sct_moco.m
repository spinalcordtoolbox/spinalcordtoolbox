function sct_moco(fname, ref)
% sct_moco(fname, ref)

tmp=sct_tempdir;
sct_gunzip(fname,tmp,'data.nii')
cd(tmp)
sct_unix(['fslroi data.nii data_ref.nii ' num2str(ref-1) ' 1'])
sct_unix(['sct_image -i data_ref.nii.gz -setorient RPI'])
dataT=sct_splitTandrename('data.nii');
for it=1:length(dataT)
    sct_unix(['sct_register_multimodal -i ' dataT{it} ' -d data_ref_RPI.nii.gz -param step=1,algo=translation,slicewise=1,metric=CC']);
    dataT{it} = [sct_tool_remove_extension(dataT{it},0) '_reg.nii.gz'];
end

sct_unix(['FSLOUTPUTTYPE=NIFTI; fslmerge -t data_reg.nii ' strjoin(dataT)])
cd ..
sct_unix(['cp ' tmp filesep 'data_reg.nii ' sct_tool_remove_extension(fname,1) '_moco.nii'])
unix(['rm -rf ' tmp])