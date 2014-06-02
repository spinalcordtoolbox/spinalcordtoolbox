% =========================================================================
% FUNCTION
% j_estimate.m
%
% Estimate beta using the General Linear Model.
%
% Shift method
% ---------------
% Possibility of using the "shift method" for (1) peak detection when X
% isn't provided, (2) finding the right HRF peak when dealing with animal
% for instance and (3) an increase of the detectability since it allows
% comparison of beta estimates within several shift process. By default no 
% shift is performed.
%
% INPUT
% Y             (1xn) vector. Data to estimate
% X             (1xn) vector. Main regressor (e.g. design convoluted with
%               hrf)
% opt           structure containing...
%  tr            time repetition (i.e. data sampling) in second. 
%                default = 1
%  nb_freq       number of discrete cosine functions used to remove low
%                frequency drifts from Y. default = 5.
%  nb_shifts      number of bi-directional shifts performed. If shift=5 then
%                10 shifts are performed. Default = 0.
% 
% OUTPUT
% results       struct
%  shift        all shift applied. Negative shift means that the regressor
%               is located "at the left" of the central position (central
%               position <=> shift=0)
%  tvalue       all tvalue regarding applied shift
%  max_tvalue   max tvalue
%  best_shift   shift applied to find max_tvalue
%
% DEPENDENCE
% spm_dctmtx
%
% COMMENT
% julien cohen-adad 2006-12-25
% =========================================================================
function results = j_estimate(Y,X,opt)



% ------------------------------------------------------------------------
%   Initializations
% ------------------------------------------------------------------------

% retreive arguments
if nargin < 3
    opt = [];
end

if isfield(opt,'tr')
    tr = opt.tr;
else
    tr = 1;
end

if isfield(opt,'nb_freq')
    nb_freq = opt.nb_freq;
else
    nb_freq = 5;
end

if isfield(opt,'nb_shifts')
    nb_shifts = opt.nb_shifts;
else
    nb_shifts = 0;
end


% misc
nb_samples = length(Y);
% clear opt

% normalize X
for i=1:size(X,1)
    Xn(i,:) = j_normalize(X(i,:));
end
clear X

% harmonize variable names
j_renvar('Xn','X');
X = X';
Y = Y';





% ------------------------------------------------------------------------
%   Remove low frequency drifts from Y by means of DCT functions
% ------------------------------------------------------------------------

% create DCT basis of regressors
D = spm_dctmtx(nb_samples,nb_freq)*sqrt(nb_samples);

% estimate l (i.e. projection of Y onto D)
l = pinv(D'*D)*D'*Y;

% calculate residuals
res_l = Y - D*l;

% reconstruct drift signal
Dl = D*l;

% reconstruct Y without low frequency drifts
Yd = Y-Dl;

% add half of Yd variance
Ydv = Yd;% + std(Yd)/2;

% harmonize variables
j_renvar('Ydv','Y');
clear Dl D l res_l Yd





% ------------------------------------------------------------------------
%   Perform estimation 
% ------------------------------------------------------------------------

j_renvar('Y','Y_noshift');
for iShift=1:nb_shifts*2+1
    index_shift = iShift-nb_shifts-1;
    % NB: when index_shift is negative, it's like Y is shifted to the left,
    % so to make things more intuitive, index_shift is multiplied by (-1)
    
    % recompose Y when shift is applied (i.e. copy signal edge to the
    % other extremity in order to avoid truncate Y)
    if index_shift<0
        Y_shift = cat(1,Y_noshift(1-index_shift:end),Y_noshift(1:-index_shift));
    elseif index_shift>0
        Y_shift = cat(1,Y_noshift(end-index_shift+1:end),Y_noshift(1:end-index_shift));
    else
        Y_shift = Y_noshift;
    end
    
    % harmonize variable
    j_renvar('Y_shift','Y');
    
    % b calculation
    b = pinv(X'*X)*X'*Y;

    % residuals calculation
    res_b = Y - X*b;

    % degre of freedom = nb_samples - (deg_regressor+1) - 2
    degf = nb_samples - 4;

    % variance of b (which is variance of residuals)
    var_b = var(res_b)/degf;
    
    % Student-test value
    tvalue = b/sqrt(var_b);
    
    % save result
    results.tvalue(iShift) = tvalue;
    results.shift(iShift) = -index_shift; % there is a minus in order to understand the shift more intuitively
end

% find max tvalue
[results.max_tvalue results.best_shift] = max(results.tvalue);
% 
% j=1;
% for i=1:size(Ybladder,1)
%     if (tvalue(i)>=6)
%         results_shift.tvalue(j) = tvalue(i);
%         results_shift.shift(j) = shift(i);
%         j=j+1;
%     end
% end
% 

% varargout{1} = results;

