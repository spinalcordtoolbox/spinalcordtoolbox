% =========================================================================
% FUNCTION
% j_1DregR
%
% One dimensional registration of 2 signals based on correlation maximization.
% For initialization purpose, At n=1, the algorithm starts at i=start_shift. Then src signal
% is shifted at i=1, then i=-1, i=2, i=-2, etc. The prior is that both signals are quite close
% to each other. This is made for a faster convergence of the algorithm.
%
% Note that a positive shift means that the signal is shifted to the right.
%
% INPUT
% src           (1,n) source signal
% dest          (1,n) destination signal
% (start_shift) integer. Start shift (default=0).
%
% OUTPUT
% src_reg       (1,n) source signal registered on destination signal
% shift         shift index to have max MI score
%
% COMMENTS
% Julien Cohen-Adad 2007-11-26
% =========================================================================
function [src_reg best_shift] = j_1DregR(src,dest,start_shift)


% parameters
th_index = 10;

% initializaton
if nargin<3, start_shift=0; end
nb_samples = length(src);
if nb_samples~=length(dest)
    error('both signals should have same length');
end
% index = 1;
R = [];
index = 1;

% compute shift index (without additional shift)
shift(index) = start_shift;

% compute shifted src signal (without additional shift)
if shift(index)>0
    src_shifted = cat(2,src(end-shift(index)+1:end),src(1:end-shift(index)));
elseif shift(index)<0
    src_shifted = cat(2,src(1-shift(index):end),src(1:-shift(index)));
else
    src_shifted = src;
end

% compute R (without additional shift)
R(index) = corr2(src_shifted,dest);
index = index + 1;

% loop on src signal permutation and estimate R
for i=1:nb_samples/2-2
    
    % compute shift index
    shift(index) = start_shift + i;
    
    % compute shifted src signal
    if shift(index)>0
        src_shifted = cat(2,src(end-shift(index)+1:end),src(1:end-shift(index)));
    elseif shift(index)<0
        src_shifted = cat(2,src(1-shift(index):end),src(1:-shift(index)));
    else
        src_shifted = src;
    end
    
    % compute R
    R(index) = corr2(src_shifted,dest);
    index = index + 1;
    
    % compute shift index, the other direction
    shift(index) = start_shift - i;
    
    % compute shifted src signal
    if shift(index)>0
        src_shifted = cat(2,src(end-shift(index)+1:end),src(1:end-shift(index)));
    elseif shift(index)<0
        src_shifted = cat(2,src(1-shift(index):end),src(1:-shift(index)));
    else
        src_shifted = src;
    end
    
    % compute R
    R(index) = corr2(src_shifted,dest);
    index = index + 1;

    % test correlation and quit loop if necessary
    [maxR index_maxR] = max(R);
    if (index-index_maxR>th_index)
        break;
    end
end

% compute reg src signal
best_shift = shift(index_maxR);
if best_shift>0
    src_reg = cat(2,src(end-best_shift+1:end),src(1:end-best_shift));
elseif best_shift<0
    src_reg = cat(2,src(1-best_shift:end),src(1:-best_shift));
else
    src_reg = src;
end
