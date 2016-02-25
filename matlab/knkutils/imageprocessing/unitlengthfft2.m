function f = unitlengthfft2(x)

% function f = unitlengthfft2(x)
%
% <x> is in the form of the magnitude output of fft2.m
%
% interpret as zero-phase spatial filter and unit-length normalize the filter.
% then put back into the Fourier domain.
%
% example:
% isequal(unitlengthfft2(10*ones(10,10)),ones(10,10))

f = abs(fft2(unitlength(real(ifft2(x)))));
