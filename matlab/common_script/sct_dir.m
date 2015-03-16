function SCT_DIR = sct_dir
% SCT_DIR = sct_dir
% Find spinalcordtoolbox directory
[~,SCT_DIR]=unix('echo ${SCT_DIR}');
sd=strsplit(SCT_DIR); sd(cellfun(@isempty,sd))=[];
SCT_DIR=sd{end};
