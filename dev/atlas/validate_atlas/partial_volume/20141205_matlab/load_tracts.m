function tracts = load_tracts(path_tracts, dim)
%
% Load tracts from the WM atlas
% Author: jcohen@polymtl.ca
% Created: 2014-12-05
%

% list all files in folder
file_tracts = dir([path_tracts, filesep, '*.nii.gz']);
nb_files = length(file_tracts)

% initialize tract
%tracts = zeros(dim(1), dim(2), dim(3), nb_files);

% loop across tracts
for i_file = 1:nb_files
    % open tract file
    file_tract = [path_tracts, filesep, file_tracts(i_file).name];
%     tracts(:, :, :, i_file) = read_avw(file_tract);
    tracts{i_file} = read_avw(file_tract);
end



