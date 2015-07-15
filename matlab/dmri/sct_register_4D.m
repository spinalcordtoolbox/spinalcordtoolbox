function sct_register_4D(data1,data2)
% sct_register_4D(data1,data2)
dbstop if error
[data1_basename,~,ext]=sct_tool_remove_extension(data1,1);

tmp_folder=sct_tempdir;

sct_gunzip(data1,tmp_folder, 'data_1.nii');
sct_gunzip(data2,tmp_folder, 'data_2.nii');

cd(tmp_folder)

sct_dmri_splitin2('data_1',1,1);
sct_dmri_splitin2('data_2',1,1);

unix(['sct_orientation -i data_1_1.nii.gz -s RPI -o data_1_1.nii.gz']);
cmd=(['sct_register_multimodal -i data_2_1.nii.gz -d data_1_1.nii.gz -p step=1,algo=slicereg2d_translation']);
sct_unix(cmd)

cmd=(['sct_apply_transfo -i data_1.nii.gz -d data_2_1.nii.gz -w ' pwd filesep 'warp_data_1_12data_2_1.nii.gz -o ' data1_basename '_reg' ext]);
sct_unix(cmd)

cd ../
unix(['rm -rf ' tmp_folder]);


