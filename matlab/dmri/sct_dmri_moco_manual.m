function sct_dmri_moco_manual(data,motionPos)
%sct_dmri_moco_manual(data,motionPos)

% change orientation and put header
nii=load_nii(data);
save_nii_v2(nii,data);
[vol1,vol2]=sct_dmri_splitin2(data,motionPos);
[vol11,vol12]=sct_dmri_splitin2(vol1,1);
unix(['rm ' vol12]);
[vol21,vol22]=sct_dmri_splitin2(vol2,1);
unix(['rm ' vol22]);

unix(['sct_orientation -i ' vol11 ' -s RPI -o ' vol11]);
unix(['sct_register_multimodal -i ' vol21 ' -d ' vol11 ' -p step=1,algo=slicereg']);
unix(['rm ' vol21 ' ' vol11]);

[~,path]=sct_tool_remove_extension(vol1,0);
unix(['sct_apply_transfo -i vol1 -d vol2 -w ' path 'warp_' sct_tool_remove_extension(vol1,0) '2' sct_tool_remove_extension(vol1,0) '.nii.gz -o /Users/taduv_admin/data/BIM/AxCaliber_1.nii.gz']); 

unix(['rm ' vol1 ' ' vol2])
%warp_AxCaliber_1_1_RPI2AxCaliber_2_1.nii.gz