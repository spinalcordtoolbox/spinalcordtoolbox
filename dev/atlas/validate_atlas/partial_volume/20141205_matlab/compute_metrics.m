%
% Batch to compute metrics using WM atlas
% Author: jcohen@polymtl.ca
% Created: 2014-12-05
%

clear all

% parameters
fname_data = 'WM_phantom_noise';
path_tracts = 'label_tracts';

% open data
data = read_avw(fname_data);
dim = size(data);

% open tracts
tracts = load_tracts(path_tracts, dim);

% estimate metrics using ML
[X_ml, std_ml, X_raw, std_raw] = m_estimate_ML_tracts(data, tracts);

X_ml

X_raw
