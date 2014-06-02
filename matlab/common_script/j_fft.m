% =========================================================================
% FUNCTION
% j_fft.m
%
% Perform FFT
% 
% INPUT
% array_in      array on which FFT will be performed
% sampling      sampling rate of array (in Hz)
% (size_window) FFT window length (in samples). Default = 512
% (freq_min)    start display at frequency sample. Default = 2
% (display_spectrum) plot result
%
% OUTPUT
% PY            power spectrum
% f             frequency sampling
% 
% COMMENT
% julien cohen-adad 2006-08-02
% =========================================================================
function varargout = j_fft(varargin)


% retreive arguments
if (nargin<2), help j_fft;, return; end
array_in = varargin{1};
sampling = varargin{2};

if (nargin<3)
    size_window = 512;
else
    size_window = varargin{3};
end

if (nargin<4)
    freq_min = 2;
else
    freq_min = varargin{4};
end

if (nargin<5)
    display_spectrum = 1;
else
    display_spectrum = varargin{5};
end


% FFT
Y = fft(array_in,size_window);

% power spectrum
PY = Y.*conj(Y)/size_window;

% convert in Hz
f = sampling*(0:size_window/2)/size_window;

% display
if display_spectrum
    figure('name','power spectrum')
    plot(f(freq_min:end),PY(freq_min:size_window/2+1))
    xlabel('frequency (in Hz)')
    zoom
end

% output
varargout{1}=PY(freq_min:size_window/2+1);
varargout{2}=f(freq_min:end);
