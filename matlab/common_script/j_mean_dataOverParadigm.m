% =========================================================================
% FUNCTION
% j_mean_dataOverParadigm
%
% Mean data over paradigm to see hemodynamic curve
%
% INPUT
% data              4d matrix. fMRI time series
% paradigm          vector with same length than size(data4d,4)
% (opt)             structure
%  linreg           do polynomial linear regression (default=1)
%  mask             3d vector, same size as data(:,:,:)
%
% OUTPUT
% data_mean
%
% COMMENTS
% julien cohen-adad 2007-04-27
% =========================================================================
function data_mean = j_mean_dataOverParadigm(data,opt)


% default initializations
if (nargin<1), help j_mean_dataOverParadigm; return; end
linreg      = 0;
mask        = Inf;

% user initialization
if ~exist('opt'), opt = []; end
if isfield(opt,'linreg'), linreg = opt.linreg; end
if isfield(opt,'mask'), mask = opt.mask; end

% polynomial detrending
if linreg
    data_d = j_detrend(data,2,mask);
end

% mean data over paradigm
[nx ny nz nt] = size(data);
size_window=300/5;
data_mean = zeros(nx,ny,nz,size_window);
for i=1:size_window:300
    data_mean = data_mean + data(:,:,:,i:i+size_window-1);
end




