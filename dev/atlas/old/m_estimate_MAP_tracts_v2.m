function [X_map,std_map,sigmaX,X_raw,std_raw] = m_estimate_MAP_tracts_v2(data,sigmaN,tracts)
% [X_map,std_map,X_raw,std_raw] = m_estimate_MAP_tracts_v2(data,sigmaN,tracts)
% 
% Partial volume correction for metric computation inside the spinal cord
% white matter tracts with maximum a posteriori estimation
% The data and the atlas must be registered before estimation
% the 3rd dimension is considered to be the z (SI) axis 
% 
% v2: new correction scheme for voxel with incomplete info
% 
% X_map: metric value estimation for each tract
% std_map: standard deviation for each tract
% 
% data: metric to quantify in the tracts 
% sigmaN: standard deviation of the gaussian noise
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

% Estimation of mean and standard deviation for a priori model
% data_nobg = data(:);
% data_nobg = data_nobg(data_nobg>0);
X0 = mean(data(:));

g2d = fspecial('gaussian',[3 3],0.5);
g1d = g2d(:,2);
g3d = zeros(3,3,3);
for k = 1:3
   g3d(:,:,k) = g1d(k) * g2d; 
end
g3d = g3d / sum(g3d(:));
data_smooth = imfilter(data,g3d);
data_smooth = data_smooth(:);
thresh = median(data_smooth(data_smooth>0)) / 100;
data_smooth = data_smooth(data_smooth>thresh);
sigmaX = std(data_smooth);

% Initializations for estimation
Y = data(:); % voxel values vector
P = zeros(length(Y),numtracts); % partial volumes matrix
for label = 1:numtracts
    P (:,label) = tracts{label}(:);
end

% Correction for incomplete information in the atlas (voxels where the
% partial volume values don't sum to 1)
% If the sum is above the correction threshold: mere normalization
% Otherwise: add extra estimations to 'absorb' the error
correc_thresh = 0.9;
number_extra = 0;
for vox = 1:size(P,1)
    vox_sum = sum(P(vox,:));
    if (vox_sum > correc_thresh)
        P(vox,:) = P(vox,:) / vox_sum;
    elseif (vox_sum > 0)
        number_extra = number_extra + 1;
        extra_col = zeros(size(P,1),1);
        extra_col(vox) = 1 - vox_sum;
        P = [P extra_col];
    end
end

numtracts_correc = size(P,2);
U_comp = ones(numtracts_correc,1);

% Maximum a posteriori estimation 
X_map = X0 + (P'*P + (sigmaN/sigmaX)*(sigmaN/sigmaX)*eye(numtracts_correc)) \ ( P'*(Y - P*(X0*U_comp)) );
X_map = X_map(1:numtracts);

% Compute standard deviation for each tract
std_map = zeros(numtracts,1);
for label = 1:numtracts
    temp = tracts{label};
    temp(temp > 0) = 1;
    temp = temp .* data;
    temp = temp(:) - X_map(label);
    temp = temp .* temp;
    std_map(label) = sqrt(sum(temp));
end

% Compute raw estimation (raw mean with partial volume weighting)
% and the associated standard deviation
X_raw = zeros(numtracts,1);
std_raw = zeros(numtracts,1);
for label = 1:numtracts
    temp = tracts{label} .* data;
    temp = temp(:);
    factors = tracts{label}(:);
    X_raw(label) = sum(temp) / sum(factors);
    std_raw(label) = std(temp);
end




end

