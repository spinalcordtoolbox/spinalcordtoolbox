function sct_dcm_dir2nii_withscaling(dcmdir)
% sct_dcm_dir2nii_withscaling('./*.dcm')
[Series, desc]=sct_dcm_dir_SeriesList(dcmdir,'Y');
Series_str=sprintf('%s ' , Series{:});
unix(['dcm2nii ' Series_str]);

for iSerie=find(~cellfun(@isempty,strfind(Series,'Scaled')))
    fnifti=dir([Series{iSerie} filesep '*.nii*']);
    if length(fnifti)==1
        j_dmri_autoscale([Series{iSerie} filesep fnifti.name],[Series{iSerie} filesep dcmdir]);
    end
end


