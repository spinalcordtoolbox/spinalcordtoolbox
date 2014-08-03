function [Series, desc]=sct_dcm_dir_SeriesList(dcmdir,varargin)
% [Series, desc] = sct_dcm_dir_SeriesList('./*.dcm' (, movefile?[Y/N]))
% Also move files to different 'run' folders if user agrees
%
% OUTPUTS:
% run. (cell): list of run name. e.g.: run{4} --> ep2d_diff_0.8mm_dirOrient_d6D40G100r20_Scaled
%
% desc. (struct): length(desc)= number of dicom in the folder
%   desc(idcm).SeriesDescription : run name
%   desc(idcm).SeriesNumber : corresponding run number
%   desc(idcm).DicomName : Dicom filename

dbstop if error
list_dcm=dir(dcmdir);
list_dcm={list_dcm.name};
list_dcm=sort_nat(list_dcm);
for idcm=1:length(list_dcm)
    dcm=dicominfo(list_dcm{idcm});
    desc(idcm).SeriesDescription=[num2str(dcm.SeriesNumber) '_' dcm.SeriesDescription(find(~isspace(dcm.SeriesDescription)))];
    desc(idcm).DicomName=list_dcm{idcm};
end
[Series,idcm,iSerie]=unique({desc.SeriesDescription});


for idcm=1:length(list_dcm)
    desc(idcm).SeriesNumber=iSerie(idcm);
end

% display
for iSerie=1:length(Series)
    disp([num2str(iSerie) ' ' Series{iSerie}])
end


% Move Files?
if isempty(varargin)
    prompt = 'Do you want to move dicoms to separate folders? Y/N [Y]: ';
    movefile = input(prompt,'s');
else
    movefile=varargin{1};
end

if isempty(movefile) || strcmp(movefile,'Y')
    for iSerie=1:length(Series)
        mkdir(Series{iSerie})
        irun_dicomlist=find(cellfun(@(x) strcmp(x,Series{iSerie}), {desc.SeriesDescription}));
        irun_dicomlist=sprintf('%s ' ,desc(irun_dicomlist).DicomName);
        unix(['mv  ' irun_dicomlist Series{iSerie} filesep]);
    end
end

