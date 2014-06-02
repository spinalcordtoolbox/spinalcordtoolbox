% =========================================================================
% FUNCTION
%
% Find peaks in a 1-D function
%
% INPUTS
% x             1-d matrix (n samples).
% (extremum)    'max', 'min', 'both' (default='max')
%
% OUTPUTS
% ind_peaks     1-d matrix. Index of all peaks.
% (val_peaks)   1-d matrix. Value of all peaks
%
% DEPENDANCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 2006-10-18
% =========================================================================
function varargout = j_findPeaks(x,opt)


% default initialization
extremum            = 'max';
show_results        = 1;
smoothing_window    = 0;

% user initialization
if (nargin<1) help j_findPeaks; return; end
if (nargin<2), opt = []; end
if isfield(opt,'extremum'), extremum = opt.extremum; end
if isfield(opt,'show_results'), show_results = opt.show_results; end
if isfield(opt,'smoothing_window'), smoothing_window = opt.smoothing_window; end

ind_peaks = [];
val_peaks = [];
n = length(x);
j = 1;

% smooth signal
if smoothing_window
    x = smooth(x,smoothing_window);
end

% find peaks
switch extremum
    case 'max'
    for i=2:n-1
        if (x(i-1)<x(i) & x(i)>=x(i+1))
            ind_peaks(j) = i;
            val_peaks(j) = x(i);
            j = j+1;
        end
    end

    case 'min'
    for i=2:n-1
        if (x(i-1)>x(i) & x(i)<x(i+1))
            ind_peaks(j) = i;
            val_peaks(j) = x(i);
            j = j+1;
        end
    end
    
    case 'both'
    for i=2:n-1
        if (x(i-1)<x(i) & x(i)>x(i+1)) | (x(i-1)>x(i) & x(i)<x(i+1))
            ind_peaks(j) = i;
            val_peaks(j) = x(i);
            j = j+1;
        end
    end
end

% show results
if show_results
    figure, plot(x,'k','Linewidth',2)
    hold on
    plot(ind_peaks,val_peaks,'ro','MarkerSize',5)
end

% output
varargout{1} = ind_peaks;
varargout{2} = val_peaks;
