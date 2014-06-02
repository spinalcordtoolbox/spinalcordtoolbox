function [ m_radius_smoothed ] = scs_smoothing(coeff_horizontal,coeff_vertical,m_radius)
% scs_radius_update 
%   This function update the radius of the spinal cord by an amount
%   determined by the gradient of the image.
%
% SYNTAX:
% [M_RADIUS_SMOOTHED] = scs_smoothing(COEFF_HORIZONTAL,COEFF_VERTICAL,M_RADIUS)
%
% _________________________________________________________________________
% INPUTS:  
%
% COEFF_HORIZONTAL
%   determines the cut-off frequency of the low-pass filter used to smooth
%   horizontaly (on one slice) the radius
%
% COEFF_VERTICAL
%   determines the cut-off frequency of the low-pass filter used to smooth
%   verticaly (from slice to slice) the radius
% 
% M_RADIUS
%   the unsmoothed radius
% _________________________________________________________________________
% OUTPUTS:
%
% M_RADIUS_SMOOTHED
%   Smoothed radius after the horizontal and vertical filtering
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[num_z num_theta]=size(m_radius);
rep=3;

% Smoothing of all the angles for each slice
Fs=num_theta;
Fpass=coeff_horizontal;
Fstop=coeff_horizontal+3;
Apass=1;
Astop=40;
match='stopband';
h = fdesign.lowpass(Fpass,Fstop,Apass,Astop,Fs);
Hd = design(h, 'cheby2', 'MatchExactly', match);

m_radius_smoothed1 = zeros(num_z,num_theta);
for i=1:num_z
    filtre = filtfilt(Hd.sosMatrix, Hd.ScaleValues, repmat(m_radius(i,:),1,rep));
    m_radius_smoothed1(i,:) = filtre(floor(rep/2)*num_theta+1:ceil(rep/2)*num_theta);
end

% Smoothing of each radius from slice to slice
Fs=num_z;
Fpass=coeff_vertical;
Fstop=coeff_vertical+3;
Apass=1;
Astop=40;
match='stopband';
h = fdesign.lowpass(Fpass,Fstop,Apass,Astop,Fs);
Hd = design(h, 'cheby2', 'MatchExactly', match);

m_radius_smoothed = zeros(num_z,num_theta);
for i=1:num_theta
    filtre = filtfilt(Hd.sosMatrix,Hd.ScaleValues, repmat(m_radius_smoothed1(:,i),rep,1));
    m_radius_smoothed(:,i) = filtre(floor(rep/2)*num_z+1:ceil(rep/2)*num_z);
end

end