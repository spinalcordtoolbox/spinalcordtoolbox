% j_convertV7toV6
%
% script that converts .mat files from matlab 7 to matlab 6
%
% INPUT 
% (-)
% 
% OUTPUT
% (-)
%
% DEPENDANCES
% j_readAnalyze()
% j_filterStruct.mat
% j_temporalFilter()
% j_writeAnalyze()
% 
% COMMENTS
% Julien Cohen-Adad 15/11/2005


% read .mat files
mat_files_V7 = spm_get(Inf,'mat','Enter .mat files to convert');

% start timer
tic
fprintf('Convert data. Please wait...');

% convert files
nb_files = size(mat_files_V7,1);
for i = 1:nb_files
    load(mat_files_V7(i,:),'mat');
    save(mat_files_V7(i,:),'mat','-V6');
end

% display elapsed time
fprintf(' OK (elapsed time %s seconds) \n',num2str(toc));
