function sct_template_extractlevels(levels)
% sct_template_extractlevels(4:-1:1)
% put everything in a folder ./template_roi
[~, SCT_DIR]=unix('echo $SCT_DIR'); SCT_DIR=SCT_DIR(1:end-1);
levels_fname=[SCT_DIR '/data/template/MNI-Poly-AMU_level.nii.gz'];

levels_template=read_avw(levels_fname);
z_lev=[];
for i=levels
    [~,~,z]=find3d(levels_template==i); z_lev(end+1)=floor(mean(z));
end
z_lev(z_lev>480)=480;

[templatelist, path]=sct_tools_ls([SCT_DIR '/data/template/MNI-Poly-AMU*']);
mkdir('template_roi')
for ifile =1:length(templatelist)
    template=load_nii([path templatelist{ifile}]);
    template_roi=template.img(:,:,z_lev);
    template_roi=make_nii(double(template_roi),[0.5 0.5 0.5],[],[]);
    save_nii_v2(template_roi,['./template_roi/' sct_tool_remove_extension(templatelist{ifile},0) '_roi'])
end

[tractlist, path]=sct_tools_ls([SCT_DIR '/data/atlas/WMtract*']);
mkdir('template_roi/atlas')
for ifile =1:length(tractlist)
    tract=load_nii([path tractlist{ifile}]);
    tract_roi=tract.img(:,:,z_lev);
    tract_roi=make_nii(double(tract_roi),[0.5 0.5 0.5],[],[]);
    save_nii_v2(tract_roi,['./template_roi/atlas/' sct_tool_remove_extension(tractlist{ifile},0) '_roi'])
end