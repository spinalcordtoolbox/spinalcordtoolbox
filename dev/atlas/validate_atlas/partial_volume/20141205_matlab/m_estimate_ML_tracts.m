function [X_ml,std_ml,X_raw,std_raw] = m_estimate_ML_tracts(data,tracts)
% Partial volume correction for metric computation inside the spinal cord
% white matter tracts with maximum likelihood estimation
% The data and the atlas must be registered before estimation
% the 3rd dimension is considered to be the z (SI) axis 
% 
% [X_ml,std_ml,X_raw,std_raw] = m_estimate_ML_tracts(data,tracts)
% X: metric value estimation for each tract
% std: standard deviation for each tract
% data: metric to quantify in the tracts 
% tracts: cell array containing the white matter atlas

[nx,ny,nz] = size(tracts{1});
numtracts = length(tracts);
[mx,my,mz] = size(data);

% Adjust data size to the atlas if necessary
if (mx >= nx)
    data = data(1:nx,:,:);
else
    temp = zeros(nx,my,mz);
    temp(1:mx,:,:) = data;
    data = temp;
end

if (my >= ny)
    data = data(:,1:ny,:);
else
    temp = zeros(nx,ny,mz);
    temp(:,1:my,:) = data;
    data = temp;
end

if (mz >= nz)
    data = data(:,:,1:nz);
else
    temp = zeros(nx,ny,nz);
    temp(:,:,1:mz) = data;
    data = temp;
end

% Compute and apply binary mask
mask = zeros(nx,ny,nz);
mask = squeeze(mask);

for label = 1:numtracts
    mask(tracts{label} > 0) = 1;
end

data = mask .* data;

% Initializations for estimation
Y = data(:); % voxel values vector
P = zeros(length(Y),numtracts); % partial volumes matrix
for label = 1:numtracts
    P (:,label) = tracts{label}(:);
end

% Maximum likelihood estimation (pseudo-inverse)
X_ml = P \ Y;

disp(['number of non-zero voxels: ', num2str(length(Y)) ])

% Compute standard deviation for each tract
std_ml = zeros(numtracts,1);
for label = 1:numtracts
    temp = tracts{label};
    temp(temp > 0) = 1;
    temp = temp .* data;
    temp = temp(:) - X_ml(label);
    temp = temp .* temp;
    std_ml(label) = sum(temp);
end

% Compute raw estimation (raw mean with partial volume weighting)
% and the associated standard deviation
X_raw = zeros(numtracts,1);
std_raw = zeros(numtracts,1);
for label = 1:numtracts
    temp = tracts{label} .* data;
    temp = temp(:);
    temp = temp(temp > 0);
    X_raw(label) = mean(temp);
    std_raw(label) = std(temp);
end




end

