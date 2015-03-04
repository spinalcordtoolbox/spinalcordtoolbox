function u = minL2TikhonovFBP(f, gamma, R, varargin)
%minL2TikhonovFBP Solves the L^2 Tikhonov problem for R being the Radon transform 
%
% Computes a minimizer of the L^2 Tikhonov functional
%
%     \| R u - f \|_2^2 + \gamma \| u \|_2^2
%
% using a filtered backprojection formula
%
% See also: minL1Tikhonov, minL2iPotts, minL2iSpars, iPottsADMM

% written by M. Storath
% $Date: 2014-06-30 11:26:34 +0200 (Mo, 30 Jun 2014) $	$Revision: 99 $




d = 1;

% Modified from iradon.m
%---------------------------------------------------------------------
len = size(f,1);
order = max(64,2^nextpow2(2*len));
n = 0:(order/2); % 'order' is always even. 
filtImpResp = zeros(1,(order/2)+1); % 'filtImpResp' is the bandlimited ramp's impulse response (values for even n are 0)
filtImpResp(1) = 1/4; % Set the DC term 
filtImpResp(2:2:end) = -1./((pi*n(2:2:end)).^2); % Set the values for odd n
filtImpResp = [filtImpResp filtImpResp(end-1:-1:2)]; 
filt = 2*real(fft(filtImpResp)); 
filt = filt(1:(order/2)+1);

filtMod = filt./(1+gamma * filt); % Tikhonov reg.
w = 2*pi*(0:size(filt,2)-1)/order;   % frequency axis up to Nyquist
filtMod(w>pi*d) = 0; 
filt = [filtMod' ; filtMod(end-1:-1:2)'];    % Symmetry of the filter
%----------------------------------------------------------------------

f(length(filt),1)=0;  % Zero pad projections


f = fft(f);    % p holds fft of projections

for i = 1:size(f,2)
    f(:,i) = f(:,i).*filt; % frequency domain filtering
end

f = real(ifft(f));     % p is the filtered projections
f(len+1:end,:) = [];   % Truncate the filtered projections



% backprojection
u = R' * f;


end

