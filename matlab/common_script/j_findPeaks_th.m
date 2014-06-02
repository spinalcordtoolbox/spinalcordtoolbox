% =========================================================================
% FUNCTION
% j_findPeaks_th
%
% Find peaks in a 1-D function based on threshold.
%
% INPUTS
% x             1-d matrix (n samples).
% threshold     integer.
% (time_window) integer. 
%
% OUTPUTS
% ind_peaks     1-d matrix. Index of all peaks.
% (val_peaks)   1-d matrix. Value of all peaks
%
% COMMENTS
% Julien Cohen-Adad 2007-10-29
% =========================================================================
function varargout = j_findPeaks_th(varargin)


% initialization
if (nargin<2) help j_findPeaks; return; end
if (nargin<3) time_window = 5; else time_window = varargin{3}; end

x = varargin{1};
th = varargin{2};

% find peaks
[ind_peaks val_peaks] = find(x>th);

% correct for multiple close values
n = length(ind_peaks);
ind_peaks_corr(1) = ind_peaks(1);
j = 2;
for i=2:n
    if ind_peaks(i)-ind_peaks(i-1)>time_window
        ind_peaks_corr(j,1) = ind_peaks(i);
        j = j+1;
    end
end

% output
varargout{1} = ind_peaks_corr;
varargout{2} = val_peaks;
