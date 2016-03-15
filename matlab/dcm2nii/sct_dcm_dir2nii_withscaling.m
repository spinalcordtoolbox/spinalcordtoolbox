function sct_dcm_dir2nii_withscaling(dcmdir)
% sct_dcm_dir2nii_withscaling('./*.dcm')
currentdirectory=pwd;
[path, name, ext]=fileparts(dcmdir); dcmdir=[name, ext];
cd(path)
[Series, desc]=sct_dcm_dir_SeriesList(dcmdir,'Y');
for i=1:length(Series)
    dicm2nii(Series{i},[Series{i} filesep]);
end

for iSerie=find(~cellfun(@isempty,strfind(Series,'Scaled')))
    fnifti=dir([Series{iSerie} filesep '*.nii*']);
    if length(fnifti)==1
        j_dmri_autoscale([Series{iSerie} filesep fnifti.name],[Series{iSerie} filesep dcmdir]);
    end
end
cd(currentdirectory)

