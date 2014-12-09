template_folder='/Volumes/users_hd2/tanguy/matlab/spinalcordtoolbox/data/template';
atlas_folder='/Volumes/users_hd2/tanguy/matlab/spinalcordtoolbox/data/atlas';
levels=5:-1:2;
template_list={'cord','CSF','GM','T2','WM'};
atlas_list=[0 1 2];

numTraxt=j_numbering(50,2,0);
% read template files
% read levels
levels_template=read_avw([template_folder filesep 'MNI-Poly-AMU_level.nii.gz']);
z_lev=[];
for i=levels
    [~,~,z]=find3d(levels_template==i); z_lev(end+1)=floor(mean(z));
end


% choose only good slices of the template
for i_f=1:length(template_list)
template=read_avw([template_folder filesep 'MNI-Poly-AMU_' template_list{i_f}]);
template_roi=template(:,:,z_lev);
save_avw_v2(template_roi(:,end:-1:1,:),['template_roi_' template_list{i_f}],'f',[0.5 0.5 0.5 1])
end

% idem for tracts
for i_f=atlas_list+1
atlas=read_avw([atlas_folder filesep 'WMtract__' numTraxt{i_f}]);
atlas_roi=atlas(:,:,z_lev);
save_avw_v2(atlas_roi(:,end:-1:1,:),['atlas_roi_' numTraxt{i_f}],'f',[0.5 0.5 0.5 1])
end